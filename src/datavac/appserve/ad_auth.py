from panel.auth import AzureAdLoginHandler, LogoutHandler, OAuthProvider, STATE_COOKIE_NAME, decode_response_body
from datavac.logging import logger

# TODO: Check if any recent Panel updates provide a way to not require this anymore

class AzureAdLogoutHandler(AzureAdLoginHandler):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    async def get(self,*args,**kwargs):
        logger.debug(f"Logging user out")
        self.set_header("Clear-Site-Data", '"cache","cookies","storage"')
        #self.redirect("/")
        self.write("Your cookies should be clear!")

def monkeypatch_oauthprovider():
    OAuthProvider.logout_handler = property(lambda self: AzureAdLogoutHandler)
    logger.debug("Oauth provider monkey-patched to support logout")
