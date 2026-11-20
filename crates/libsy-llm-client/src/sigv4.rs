// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AWS SigV4 request signing, used to call Amazon Bedrock with IAM credentials.
//!
//! Credentials come from the AWS credential provider chain and are fetched again before they
//! expire, so a long-running router needs no token refresh or restart.
//!
//! The signature covers the method, path, query, host, signed headers and a hash of the body,
//! so signing applies to a built `reqwest::Request`, not to a `RequestBuilder`. The caller
//! sends the signed request with `Client::execute`. Streamed responses need nothing extra,
//! because the request body is complete before the call.

use std::time::{Duration, SystemTime};

use aws_config::{BehaviorVersion, SdkConfig};
use aws_credential_types::Credentials;
use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign as sigv4_sign};
use aws_sigv4::sign::v4;
use tokio::sync::{Mutex, OnceCell};

use crate::backend::HttpBackendConfig;
use crate::error::{LlmClientError, Result};

/// Default signing service name.
pub const DEFAULT_SERVICE: &str = "bedrock";

/// The service and region to sign for.
///
/// When `region` is unset, the request is signed for the region in its host, such as
/// `us-east-1` in `bedrock-runtime.us-east-1.amazonaws.com`, because AWS rejects a signature
/// for any other region. A host outside AWS, or one without a region, uses the provider
/// chain's region (`AWS_REGION`, the profile, or instance metadata).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SigV4Settings {
    pub service: String,
    pub region: Option<String>,
}

impl SigV4Settings {
    #[must_use]
    pub fn new(service: impl Into<String>, region: Option<String>) -> Self {
        Self {
            service: service.into(),
            region,
        }
    }
}

/// The AWS credential provider chain and a cache of its credentials.
///
/// The chain is loaded on the first signed call, not at config load: loading is async, and a
/// router should start even where no credentials exist yet. The first call then fails with
/// the cause. The same load supplies the fallback region.
///
/// aws-config's provider chain does not cache credentials (SDK service clients do), so this
/// type caches them. Otherwise every request would read the profile files or call the
/// container endpoint, instance metadata or STS again.
#[derive(Debug, Default)]
pub struct CredentialSource {
    config: OnceCell<SdkConfig>,
    // The last credentials and when to fetch them again. One lock, so one fetch at a time.
    cached: Mutex<Option<(Credentials, SystemTime)>>,
}

// The timing of the AWS SDK clients' own credential cache (`IdentityCache::lazy`): fetch
// again 10 seconds before credentials expire, and after 15 minutes for credentials that state
// no expiry, such as keys in a profile file that another tool rewrites.
const EXPIRY_BUFFER: Duration = Duration::from_secs(10);
const NO_EXPIRY_LIFETIME: Duration = Duration::from_secs(15 * 60);

fn refresh_time(expiry: Option<SystemTime>, now: SystemTime) -> SystemTime {
    match expiry {
        Some(expiry) => expiry.checked_sub(EXPIRY_BUFFER).unwrap_or(expiry),
        None => now + NO_EXPIRY_LIFETIME,
    }
}

impl CredentialSource {
    async fn config(&self) -> &SdkConfig {
        self.config
            .get_or_init(|| aws_config::load_defaults(BehaviorVersion::latest()))
            .await
    }

    async fn provider(&self) -> Result<SharedCredentialsProvider> {
        self.config()
            .await
            .credentials_provider()
            .ok_or_else(|| LlmClientError::Configuration {
                message: "sigv4 auth is configured but no AWS credentials provider could be \
                          resolved (checked environment, profile and container/instance \
                          metadata)"
                    .to_string(),
            })
    }

    async fn credentials(&self) -> Result<Credentials> {
        let mut cached = self.cached.lock().await;
        let now = SystemTime::now();
        if let Some((credentials, refresh_at)) = cached.as_ref()
            && now < *refresh_at
        {
            return Ok(credentials.clone());
        }
        let credentials = self
            .provider()
            .await?
            .provide_credentials()
            .await
            .map_err(credentials_error)?;
        *cached = Some((credentials.clone(), refresh_time(credentials.expiry(), now)));
        Ok(credentials)
    }

    /// The region the provider chain resolved. Only consulted when neither the client nor
    /// the host names one.
    async fn chain_region(&self) -> Option<String> {
        self.config().await.region().map(ToString::to_string)
    }

    // A source that skips the environment: `provider` and `region` stand in for the chain.
    #[cfg(test)]
    pub(crate) fn for_tests(
        provider: impl ProvideCredentials + 'static,
        region: Option<&str>,
    ) -> Self {
        let config = SdkConfig::builder()
            .credentials_provider(SharedCredentialsProvider::new(provider))
            .region(region.map(|region| aws_config::Region::new(region.to_string())))
            .build();
        let source = Self::default();
        source.config.set(config).expect("fresh cell");
        source
    }
}

/// Signs `request` in place with SigV4.
///
/// Returns an error rather than leaving the request unsigned, which the provider would answer
/// with HTTP 401.
pub async fn sign_request(
    request: &mut reqwest::Request,
    settings: &SigV4Settings,
    source: &CredentialSource,
) -> Result<()> {
    let credentials = source.credentials().await?;

    let host_region = request.url().host_str().and_then(region_from_host);
    let region = match (settings.region.clone(), host_region) {
        (Some(region), _) => region,
        (None, Some(region)) => region.to_string(),
        (None, None) => {
            source
                .chain_region()
                .await
                .ok_or_else(|| LlmClientError::Configuration {
                    message: "sigv4 auth needs a region: set it on the llm client, use a \
                              base_url whose host names one, or set it in the environment"
                        .to_string(),
                })?
        }
    };

    let identity = credentials.into();
    let signing_settings = SigningSettings::default();
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(&region)
        .name(&settings.service)
        .time(SystemTime::now())
        .settings(signing_settings)
        .build()
        .map_err(|error| LlmClientError::Configuration {
            message: format!("sigv4 signing params: {error}"),
        })?;

    // The signature covers a hash of the body, so the body must be in memory. Upstream
    // requests are a GET without a body (`/props`) or buffered JSON. A streaming body is
    // refused.
    let body_bytes = match request.body() {
        None => &[][..],
        Some(body) => body
            .as_bytes()
            .ok_or_else(|| LlmClientError::Configuration {
                message: "sigv4 auth cannot sign a streaming request body".to_string(),
            })?,
    };

    // SigV4 signs only the headers it lists in `SignedHeaders`, and AWS accepts unsigned
    // extra headers. A value that is not visible ASCII cannot be canonicalized, so it is
    // sent unsigned rather than failing the request.
    let headers: Vec<(&str, &str)> = request
        .headers()
        .iter()
        .filter_map(|(name, value)| value.to_str().ok().map(|value| (name.as_str(), value)))
        .collect();

    let signable = SignableRequest::new(
        request.method().as_str(),
        request.url().as_str(),
        headers.into_iter(),
        SignableBody::Bytes(body_bytes),
    )
    .map_err(|error| LlmClientError::Configuration {
        message: format!("sigv4 signable request: {error}"),
    })?;

    let (instructions, _signature) = sigv4_sign(signable, &params.into())
        .map_err(|error| LlmClientError::Configuration {
            message: format!("sigv4 signing failed: {error}"),
        })?
        .into_parts();

    for (name, value) in instructions.headers() {
        let header_name = reqwest::header::HeaderName::try_from(name).map_err(|error| {
            LlmClientError::Configuration {
                message: format!("sigv4 header name: {error}"),
            }
        })?;
        let mut header_value = reqwest::header::HeaderValue::from_str(value).map_err(|error| {
            LlmClientError::Configuration {
                message: format!("sigv4 header value: {error}"),
            }
        })?;
        // The signature and the session token stay out of logs, as the other auth modes'
        // secrets do. The date and payload hash stay visible.
        header_value.set_sensitive(
            header_name == reqwest::header::AUTHORIZATION || header_name == "x-amz-security-token",
        );
        request.headers_mut().insert(header_name, header_value);
    }
    Ok(())
}

/// Rejects SigV4 settings that could send a signature or another credential to the wrong host.
///
/// AWS accepts a signed request for about 15 minutes, so anyone who reads one can send it
/// again in that time. Signing keeps the headers it does not set, so a key from another auth
/// mode would be sent along.
pub(crate) fn validate(config: &HttpBackendConfig, model_name: &str) -> Result<()> {
    let error = |message: String| LlmClientError::Configuration {
        message: format!("model {model_name:?} {message}"),
    };
    if config.api_key.is_some() || config.forward_auth {
        return Err(error(
            "combines sigv4 with api_key or forward_auth; set one auth mode".to_string(),
        ));
    }
    // Plain HTTP stays allowed for loopback, where test servers run.
    let url = reqwest::Url::parse(&config.base_url)
        .ok()
        .filter(|url| url.scheme() == "https" || is_loopback(url))
        .ok_or_else(|| error("needs an https base_url for sigv4".to_string()))?;
    if let Some(name) = config.extra_headers.keys().find(|name| {
        let name = name.to_ascii_lowercase();
        name == "authorization" || name == "host" || name.starts_with("x-amz-")
    }) {
        return Err(error(format!(
            "extra_headers cannot set {name:?} with sigv4; the signer owns it"
        )));
    }
    let host = url.host_str().unwrap_or_default();
    if !is_aws_host(host) && !is_loopback(&url) {
        tracing::warn!(
            model = model_name,
            host,
            "sigv4 sends signed requests to a host outside AWS"
        );
    }
    Ok(())
}

fn is_loopback(url: &reqwest::Url) -> bool {
    match url.host_str() {
        Some("localhost") => true,
        Some(host) => host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback()),
        None => false,
    }
}

fn is_aws_host(host: &str) -> bool {
    [".amazonaws.com", ".amazonaws.com.cn", ".api.aws"]
        .iter()
        .any(|suffix| host.ends_with(suffix))
}

// A timeout or an error inside a credential source, such as instance metadata that briefly
// stops answering, may clear on the next attempt, so it is a transport error, which the
// client retries. Missing or invalid credentials fail at once as configuration errors.
fn credentials_error(error: CredentialsError) -> LlmClientError {
    let message = format!("could not resolve AWS credentials for sigv4: {error}");
    match error {
        CredentialsError::ProviderTimedOut(_) | CredentialsError::ProviderError(_) => {
            LlmClientError::Transport {
                source: message.into(),
            }
        }
        _ => LlmClientError::Configuration { message },
    }
}

/// The AWS region named in an endpoint host, such as `us-east-1` in
/// `bedrock-runtime.us-east-1.amazonaws.com`, `bedrock-runtime-fips.us-east-1.amazonaws.com`, or a
/// VPC endpoint under `bedrock-runtime.us-east-1.vpce.amazonaws.com`. Hosts outside AWS have
/// none, even when a label looks like a region.
fn region_from_host(host: &str) -> Option<&str> {
    if !is_aws_host(host) {
        return None;
    }
    host.split('.').find(|label| {
        let parts: Vec<&str> = label.split('-').collect();
        let (Some(first), Some(last)) = (parts.first(), parts.last()) else {
            return false;
        };
        parts.len() >= 3
            && first.len() == 2
            && parts[..parts.len() - 1]
                .iter()
                .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_lowercase()))
            && !last.is_empty()
            && last.bytes().all(|b| b.is_ascii_digit())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_credential_types::Credentials;

    // Temporary credentials with a session token, like those of a role. `for_tests` has no
    // session token.
    fn fixed_source() -> CredentialSource {
        source_with(
            Credentials::for_tests_with_session_token(),
            Some("us-east-1"),
        )
    }

    fn source_with(
        provider: impl ProvideCredentials + 'static,
        region: Option<&str>,
    ) -> CredentialSource {
        CredentialSource::for_tests(provider, region)
    }

    fn request(body: &str) -> reqwest::Request {
        request_to(
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
            body,
        )
    }

    fn request_to(url: &str, body: &str) -> reqwest::Request {
        reqwest::Client::new()
            .post(url)
            .header("content-type", "application/json")
            .body(body.to_string())
            .build()
            .expect("buildable")
    }

    #[test]
    fn a_default_async_sleep_is_available() {
        // The other tests use a fixed provider. The real chain retries, and its providers panic
        // without a default async sleep, which aws-config's `rt-tokio` feature provides.
        assert!(
            aws_smithy_async::rt::sleep::default_async_sleep().is_some(),
            "no default AsyncSleep: enable aws-config's `rt-tokio` feature"
        );
    }

    #[tokio::test]
    async fn signs_with_the_expected_headers() {
        let mut req = request(r#"{"model":"m"}"#);
        sign_request(
            &mut req,
            &SigV4Settings::new(DEFAULT_SERVICE, Some("us-east-1".to_string())),
            &fixed_source(),
        )
        .await
        .expect("signs");

        let auth = req
            .headers()
            .get(reqwest::header::AUTHORIZATION)
            .expect("authorization present");
        assert!(auth.is_sensitive(), "signature must not be logged");
        let auth = String::from_utf8_lossy(auth.as_bytes()).to_string();
        assert!(auth.starts_with("AWS4-HMAC-SHA256"), "got {auth}");
        assert!(
            auth.contains("/us-east-1/bedrock/aws4_request"),
            "got {auth}"
        );
        let date = req.headers().get("x-amz-date").expect("date present");
        assert!(!date.is_sensitive(), "the date is not secret");
        // Temporary credentials add their session token to the request.
        let token = req
            .headers()
            .get("x-amz-security-token")
            .expect("session token present");
        assert!(token.is_sensitive(), "session token must not be logged");
    }

    #[tokio::test]
    async fn the_signature_covers_the_body() {
        let settings = SigV4Settings::new(DEFAULT_SERVICE, Some("us-east-1".to_string()));
        let source = fixed_source();

        let mut one = request(r#"{"model":"a"}"#);
        let mut two = request(r#"{"model":"b"}"#);
        sign_request(&mut one, &settings, &source)
            .await
            .expect("signs");
        sign_request(&mut two, &settings, &source)
            .await
            .expect("signs");

        // The signature depends on the body.
        assert_ne!(
            one.headers().get(reqwest::header::AUTHORIZATION),
            two.headers().get(reqwest::header::AUTHORIZATION),
        );
    }

    #[tokio::test]
    async fn a_missing_region_is_an_error_not_an_unsigned_send() {
        let source = source_with(Credentials::for_tests_with_session_token(), None);

        // A host without a region, so neither the client, the host nor the environment has one.
        let mut req = request_to("https://gateway.example.test/v1/chat/completions", "{}");
        let error = sign_request(
            &mut req,
            &SigV4Settings::new(DEFAULT_SERVICE, None),
            &source,
        )
        .await
        .expect_err("must refuse");
        assert!(format!("{error}").contains("region"), "got {error}");
        assert!(!req.headers().contains_key(reqwest::header::AUTHORIZATION));
    }

    #[tokio::test]
    async fn a_request_without_a_body_is_signed() {
        // The llama.cpp `/props` probe is a GET with no body.
        let mut req = reqwest::Client::new()
            .get("https://bedrock-runtime.us-east-1.amazonaws.com/props")
            .build()
            .expect("buildable");
        sign_request(
            &mut req,
            &SigV4Settings::new(DEFAULT_SERVICE, None),
            &fixed_source(),
        )
        .await
        .expect("signs");
        assert!(req.headers().contains_key(reqwest::header::AUTHORIZATION));
    }

    #[derive(Debug)]
    struct FailingProvider(fn() -> CredentialsError);

    impl ProvideCredentials for FailingProvider {
        fn provide_credentials<'a>(
            &'a self,
        ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            aws_credential_types::provider::future::ProvideCredentials::ready(Err((self.0)()))
        }
    }

    #[tokio::test]
    async fn only_credential_timeouts_and_source_errors_are_transport_errors() {
        // The client retries transport errors, so these decide which failures get another try.
        let cases: [(fn() -> CredentialsError, bool); 3] = [
            (
                || CredentialsError::provider_timed_out(Duration::from_secs(1)),
                true,
            ),
            (
                || CredentialsError::provider_error("metadata service returned 503"),
                true,
            ),
            (CredentialsError::not_loaded_no_source, false),
        ];
        for (failure, retryable) in cases {
            let source = source_with(FailingProvider(failure), None);
            let mut req = request("{}");
            let error = sign_request(
                &mut req,
                &SigV4Settings::new(DEFAULT_SERVICE, None),
                &source,
            )
            .await
            .expect_err("no credentials");
            assert_eq!(
                matches!(error, LlmClientError::Transport { .. }),
                retryable,
                "got {error:?}"
            );
            assert!(
                format!("{error}").contains("AWS credentials"),
                "got {error}"
            );
            assert!(!req.headers().contains_key(reqwest::header::AUTHORIZATION));
        }
    }

    #[derive(Debug)]
    struct CountingProvider {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        expiry: Option<SystemTime>,
    }

    impl ProvideCredentials for CountingProvider {
        fn provide_credentials<'a>(
            &'a self,
        ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            aws_credential_types::provider::future::ProvideCredentials::ready(Ok(Credentials::new(
                "AKIDEXAMPLE",
                "secret",
                Some("token".to_string()),
                self.expiry,
                "counting",
            )))
        }
    }

    #[tokio::test]
    async fn credentials_are_resolved_again_only_near_expiry() {
        // Without the cache, every signed request would call the provider again.
        let now = SystemTime::now();
        for (expiry, resolutions) in [
            (None, 1),
            (Some(now + Duration::from_secs(3600)), 1),
            (Some(now + Duration::from_secs(5)), 3),
        ] {
            let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let source = source_with(
                CountingProvider {
                    calls: std::sync::Arc::clone(&calls),
                    expiry,
                },
                Some("us-east-1"),
            );
            for _ in 0..3 {
                sign_request(
                    &mut request("{}"),
                    &SigV4Settings::new(DEFAULT_SERVICE, None),
                    &source,
                )
                .await
                .expect("signs");
            }
            assert_eq!(
                calls.load(std::sync::atomic::Ordering::SeqCst),
                resolutions,
                "expiry {expiry:?}"
            );
        }
    }

    #[test]
    fn credentials_are_kept_until_shortly_before_they_expire() {
        let now = SystemTime::now();
        let expiry = now + Duration::from_secs(3600);
        assert_eq!(refresh_time(Some(expiry), now), expiry - EXPIRY_BUFFER);
        assert_eq!(refresh_time(None, now), now + NO_EXPIRY_LIFETIME);
        // Credentials at the end of their life are fetched again on the next request.
        assert!(refresh_time(Some(now), now) < now);
    }

    #[test]
    fn finds_the_region_in_aws_endpoint_hosts() {
        for (host, region) in [
            ("bedrock-runtime.us-east-1.amazonaws.com", Some("us-east-1")),
            (
                "bedrock-runtime-fips.us-gov-west-1.amazonaws.com",
                Some("us-gov-west-1"),
            ),
            (
                "vpce-0a1b2c-x1y2.bedrock-runtime.eu-central-2.vpce.amazonaws.com",
                Some("eu-central-2"),
            ),
            (
                "bedrock-mantle.ap-southeast-4.api.aws",
                Some("ap-southeast-4"),
            ),
            ("gateway.example.test", None),
            ("my-proxy-1.example.com", None),
            ("localhost", None),
        ] {
            assert_eq!(region_from_host(host), region, "{host}");
        }
    }

    const BEDROCK_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1";

    fn backend_config(base_url: &str) -> HttpBackendConfig {
        HttpBackendConfig {
            base_url: base_url.to_string(),
            api_key: None,
            forward_auth: false,
            extra_headers: std::collections::BTreeMap::new(),
            extra_body: std::collections::BTreeMap::new(),
            omit_body_fields: std::collections::BTreeSet::new(),
            reasoning_effort: None,
            max_retries: 0,
            failure_cooldown: std::time::Duration::ZERO,
            timeout: None,
            sigv4: Some(SigV4Settings::new(DEFAULT_SERVICE, None)),
        }
    }

    #[test]
    fn sigv4_is_the_only_auth_mode() {
        let mut with_key = backend_config(BEDROCK_URL);
        with_key.api_key = Some("key".to_string());
        let mut forwarding = backend_config(BEDROCK_URL);
        forwarding.forward_auth = true;
        for config in [with_key, forwarding] {
            let error = validate(&config, "m").expect_err("must refuse");
            assert!(format!("{error}").contains("one auth mode"), "got {error}");
        }
        validate(&backend_config(BEDROCK_URL), "m").expect("sigv4 alone is valid");
    }

    #[test]
    fn a_signed_request_needs_https_except_on_loopback() {
        for (base_url, allowed) in [
            (BEDROCK_URL, true),
            ("https://gateway.example.test/v1", true),
            ("http://127.0.0.1:8080/v1", true),
            ("http://localhost:8080/v1", true),
            ("http://[::1]:8080/v1", true),
            (
                "http://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
                false,
            ),
            ("http://gateway.example.test/v1", false),
            ("not a URL", false),
        ] {
            assert_eq!(
                validate(&backend_config(base_url), "m").is_ok(),
                allowed,
                "{base_url}"
            );
        }
    }

    #[test]
    fn extra_headers_cannot_set_what_the_signer_sets() {
        for name in [
            "Authorization",
            "host",
            "X-Amz-Date",
            "x-amz-security-token",
            "x-amz-content-sha256",
        ] {
            let mut config = backend_config(BEDROCK_URL);
            config
                .extra_headers
                .insert(name.to_string(), "value".to_string());
            let error = validate(&config, "m").expect_err(name);
            assert!(format!("{error}").contains(name), "got {error}");
        }
        let mut config = backend_config(BEDROCK_URL);
        config
            .extra_headers
            .insert("x-team".to_string(), "value".to_string());
        validate(&config, "m").expect("other headers are allowed");
    }

    #[tokio::test]
    async fn an_unset_region_signs_for_the_hosts_region_not_the_environments() {
        // The environment says us-east-1 and the endpoint is in us-west-2. Bedrock rejects a
        // signature for any region but the endpoint's ("Credential should be scoped to a
        // valid region"), so the host's region is used.
        let mut req = request_to(
            "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
            r#"{"model":"m"}"#,
        );
        sign_request(
            &mut req,
            &SigV4Settings::new(DEFAULT_SERVICE, None),
            &fixed_source(),
        )
        .await
        .expect("signs");
        let auth = req
            .headers()
            .get(reqwest::header::AUTHORIZATION)
            .expect("signed");
        let auth = String::from_utf8_lossy(auth.as_bytes()).to_string();
        assert!(
            auth.contains("/us-west-2/bedrock/aws4_request"),
            "got {auth}"
        );
    }

    #[tokio::test]
    async fn a_configured_region_wins_over_the_host() {
        let mut req = request_to(
            "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
            r#"{"model":"m"}"#,
        );
        let settings = SigV4Settings::new(DEFAULT_SERVICE, Some("eu-west-1".to_string()));
        sign_request(&mut req, &settings, &fixed_source())
            .await
            .expect("signs");
        let auth = req
            .headers()
            .get(reqwest::header::AUTHORIZATION)
            .expect("signed");
        let auth = String::from_utf8_lossy(auth.as_bytes()).to_string();
        assert!(
            auth.contains("/eu-west-1/bedrock/aws4_request"),
            "got {auth}"
        );
    }
}
