from nanobot.channels.weixin import WeixinChannel, WeixinConfig


def test_weixin_local_file_fallback_is_disabled_by_default() -> None:
    config = WeixinConfig()

    assert config.local_file_fallback_dirs == []


def test_weixin_media_download_defaults_are_conservative() -> None:
    config = WeixinConfig()

    assert config.media_download_max_attempts == 2
    assert config.media_download_concurrency == 1
    assert config.media_download_max_bytes == 128 * 1024 * 1024
    assert config.media_download_trust_env is False


def test_weixin_media_download_accepts_camel_case_config() -> None:
    config = WeixinConfig(
        mediaDownloadMaxAttempts=3,
        mediaDownloadConcurrency=2,
        mediaDownloadMaxBytes=1024,
        mediaDownloadTrustEnv=True,
    )

    assert config.media_download_max_attempts == 3
    assert config.media_download_concurrency == 2
    assert config.media_download_max_bytes == 1024
    assert config.media_download_trust_env is True


def test_weixin_media_download_max_bytes_can_be_disabled() -> None:
    config = WeixinConfig(mediaDownloadMaxBytes=None)

    assert config.media_download_max_bytes is None


def test_weixin_proxy_redaction_hides_credentials() -> None:
    assert (
        WeixinChannel._redact_proxy_value("http://user:secret@127.0.0.1:7890")
        == "http://***@127.0.0.1:7890"
    )
    assert WeixinChannel._redact_proxy_value("http://127.0.0.1:7890") == "http://127.0.0.1:7890"
