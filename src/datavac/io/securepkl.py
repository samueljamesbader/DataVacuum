from pickle import dumps as insecure_dump_bytes, loads as insecure_load_bytes
import os
import hmac
import hashlib
import aiofiles
from pathlib import Path
from datavac.util.logging import logger

# See https://www.synopsys.com/blogs/software-security/python-pickling/
# for "secure" use of pickle
# And https://docs.python.org/3/library/pickle.html#pickle-restrict
# for restricting pickle

class SecurePkl:
    CACHE_DIR = os.environ['DATAVACUUM_CACHE_DIR']
    CHECK_CACHE_SIG = not (os.environ.get('DATAVACUUM_CHECK_CACHE_SIG','True').lower() == 'false')

    def __init__(self):
        self.dk: bytes = os.environ['DATAVACUUM_PICKLE_SECRET'].encode('ascii')
        self.DIGEST_SIZE: int = hmac.new(self.dk, None, hashlib.sha256).digest_size

    def secure_dumps(self,obj):
        pickled_data=insecure_dump_bytes(obj)
        dig=hmac.new(self.dk, pickled_data, hashlib.sha256).digest()
        return dig+pickled_data
    def secure_filedump(self,obj, file):
        with open(file,'wb') as f:
            f.write(self.secure_dumps(obj))
    async def secure_filedump_async(self,obj, file):
        async with aiofiles.open(file,'wb') as f:
            await f.write(self.secure_dumps(obj))

    def secure_loads(self,b):
        read_dig,pickled_data=b[:self.DIGEST_SIZE],b[self.DIGEST_SIZE:]
        comp_dig=hmac.digest(self.dk, pickled_data, hashlib.sha256)
        if not hmac.compare_digest(read_dig,comp_dig):
            raise Exception(f"Signature check failed, cannot securely load pickle.")
        data=insecure_load_bytes(pickled_data)
        return data

    def is_in_local_cache(self, file):
        try:
            return Path(file).relative_to(self.CACHE_DIR)
        except:
            return False

    def secure_fileload(self,file):
        logger.info(f"Reading {str(file)}")
        with open(file,'rb') as f:
            msg=f.read()

        # If the file is in the local CACHE_DIR, we can skip the security check for speed
        # The security check is for files on DataGrove where other users have access.
        if not self.CHECK_CACHE_SIG and self.is_in_local_cache(file):
            return insecure_load_bytes(msg[self.DIGEST_SIZE:])
        else:
            try:
                return self.secure_loads(msg)
            except Exception as e:
                if 'Signature check failed' in str(e):
                    raise Exception(f"Signature check failed for {str(file)}, cannot securely load pickle.")
                else:
                    raise e

    async def secure_fileload_async(self,file):
        logger.info("Reading")
        async with aiofiles.open(file,'rb') as f:
            msg=await f.read()
        try:
            return self.secure_loads(msg)
        except Exception as e:
            if 'Signature check failed' in str(e):
                raise Exception(f"Signature check failed for {str(file)}, cannot securely load pickle.")
            else:
                raise e