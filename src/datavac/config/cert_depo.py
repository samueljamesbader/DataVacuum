from typing import Optional


class CertDepo():
    def get_ssl_rootcert_path_for_db(self) -> Optional[str]:
        """Returns the path to the SSL root certificate for communicating to the database, or None if not relevant.
        
        Default behavior is to return None, indicating that no SSL root certificate is required.
        Subclasses should override this method to provide a specific path (and/or ensure download) if needed.
        """
        return None
    
    def get_ssl_rootcert_path_for_deployment(self) -> Optional[str]:
        """Returns the path to the SSL root certificate for communicating to the deployment, or None if not relevant.
        
        Default behavior is to return None, indicating that no SSL root certificate is required.
        Subclasses should override this method to provide a specific path (and/or ensure download) if needed.
        """
        return None

    def get_ssl_rootcert_path_for_vault(self) -> Optional[str]:
        """Returns the path to the SSL root certificate for communicating to the vault, or None if not relevant.
        
        Default behavior is to return None, indicating that no SSL root certificate is required.
        Subclasses should override this method to provide a specific path (and/or ensure download) if needed.
        """
        return None