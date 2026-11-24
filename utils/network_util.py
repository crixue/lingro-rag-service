import os

from configs.model_config import VPN_PROXY_HTTP_URL


def request_vpn_proxies() -> dict:
    '''
    :return: proxies
    '''
    proxies = {'http': VPN_PROXY_HTTP_URL, 'https': VPN_PROXY_HTTP_URL}
    return proxies

class VPNContextManager:

    def _setup_vpn_proxy(self) -> None:
        """
        Sets up the VPN proxy for the Google GenAI API.

        This function sets the `http_proxy` and `https_proxy` environment variables
        to the VPN proxy URL. This is necessary for the Google GenAI API to work
        correctly.

        Raises:
            ValueError: If the VPN proxy URL is not set in the environment.
        """
        proxy = VPN_PROXY_HTTP_URL
        if not proxy:
            raise ValueError("VPN proxy URL is not set in the environment.")
        os.environ['http_proxy'] = proxy
        os.environ['HTTP_PROXY'] = proxy
        os.environ['https_proxy'] = proxy
        os.environ['HTTPS_PROXY'] = proxy

    def _remove_vpn_proxy(self) -> None:
        """
        Removes the VPN proxy settings from the environment.

        This function removes the `http_proxy` and `https_proxy` environment
        variables from the current process. This is useful for cleaning up the
        environment after using the VPN proxy.
        """
        os.environ.pop('http_proxy', None)
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('HTTPS_PROXY', None)

    def __enter__(self):
        self._setup_vpn_proxy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_vpn_proxy()
        return False