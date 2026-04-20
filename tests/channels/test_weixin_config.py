from nanobot.channels.weixin import WeixinConfig


def test_weixin_local_file_fallback_is_disabled_by_default() -> None:
    config = WeixinConfig()

    assert config.local_file_fallback_dirs == []
