from abc import abstractmethod, ABC
from typing import Any
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import aiohttp
import asyncio
from datetime import datetime
import json
import logging
from ratelimiter import RateLimiter

from open_biomed.data import Molecule, Protein

class DBRequester(ABC):
    def __init__(self, db_url: str=None, timeout: int=30) -> None:
        self.db_url = db_url
        self.timeout = timeout

    @RateLimiter(max_calls=5, period=1)
    async def run(self, accession: str="") -> str:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(self.db_url.format(accession=accession)) as response:
                    if response.status == 200:
                        content = await response.read()
                        content = content.decode("utf-8")
                        logging.info("Downloaded results successfully")
                    else:
                        logging.warning(f"HTTP request failed, status {response.status}")
                        raise Exception()
        except Exception as e:
            content = None
            logging.error(f"Download failed. Exception: {e}")
            raise e
        return self._parse_and_save_outputs(content)

    @abstractmethod
    def _parse_and_save_outputs(self, outputs: str="") -> str:
        # Parse the outputs and save them at a local file
        raise NotImplementedError

class UniProtRequester(DBRequester):
    def __init__(self, 
        db_url: str="https://rest.uniprot.org/uniprotkb/{accession}?format=json", 
        timeout: int=30
    ) -> None:
        super().__init__(db_url, timeout)

    def _parse_and_save_outputs(self, outputs: str="") -> str:
        obj = json.loads(outputs)
        protein = Protein.from_fasta(obj["sequence"]["value"])
        timestamp = int(datetime.now().timestamp() * 1000)
        pkl_file = f"./tmp/protein_{timestamp}.pkl"
        protein.save_binary(pkl_file)
        return pkl_file

class PDBRequester(DBRequester):
    def __init__(self, 
        db_url: str="https://files.rcsb.org/download/{accession}.pdb", 
        timeout: int=30
    ) -> None:
        super().__init__(db_url, timeout)

    def _parse_and_save_outputs(self, outputs: str="") -> str:
        timestamp = int(datetime.now().timestamp() * 1000)
        pdb_file = f"./tmp/protein_{timestamp}.pdb"
        with open(pdb_file, "w") as f:
            f.write(outputs)
        protein = Protein.from_pdb_file(pdb_file)
        pkl_file = f"./tmp/protein_{timestamp}.pkl"
        protein.save_binary(pkl_file)
        protein.save_pdb(pdb_file)
        return pkl_file

if __name__ == "__main__":
    requester = UniProtRequester()
    asyncio.run(requester.run("P0DTC2"))

    requester = PDBRequester()
    asyncio.run(requester.run("6LVN"))