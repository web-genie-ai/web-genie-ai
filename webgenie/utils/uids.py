import bittensor as bt
import random
import re
import numpy as np
from typing import List, Tuple
from webgenie.constants import (
    VPERMIT_TAO_LIMIT,
)


def is_validator(metagraph: "bt.metagraph.Metagraph", uid: int) -> bool:
    return metagraph.S[uid] > VPERMIT_TAO_LIMIT


def get_validator_index(self, uid: int) -> Tuple[int, int]:
    validator_uids = []
    for each_uid in range(self.metagraph.n.item()):
        if is_validator(self.metagraph, each_uid):
            validator_uids.append(each_uid)  
    validator_uids.sort(key=lambda uid: self.metagraph.S[uid], reverse=True)
    try:
        return validator_uids.index(uid), len(validator_uids)
    except ValueError:
        return -1, len(validator_uids)


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    # if not metagraph.axons[uid].is_serving:
    #     return False
    # Filter validator permit > 1024 stake.
    if metagraph.S[uid] > VPERMIT_TAO_LIMIT:
        return False
    # Available otherwise.
    return True


def get_most_available_uid(self, exclude: List[int] = None) -> int:
    """Returns the most available uid from the metagraph.
    Returns:
        uid (int): Most available uid.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    
    return candidate_uids[np.argmax(self.metagraph.I[candidate_uids])]


def get_all_available_uids(
    self, exclude: List[int] = None
) -> np.ndarray:
    ip_count = {}
    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(self.metagraph, uid) 
        if not uid_is_available:
            continue
        uid_is_not_excluded = exclude is None or uid not in exclude
        if not uid_is_not_excluded:
            continue
        ip = self.metagraph.addresses[uid]
        ip = ip.split(":")[0]
        ip_count[ip] = ip_count[ip] + 1 if ip in ip_count else 1
    
    avail_uids = []
    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(self.metagraph, uid) 
        if not uid_is_available:
            continue
        uid_is_not_excluded = exclude is None or uid not in exclude
        if not uid_is_not_excluded:
            continue
        ip = self.metagraph.addresses[uid]
        ip = ip.split(":")[0]
        has_too_many_ips = ip_count[ip] > 3
        if not has_too_many_ips:
            avail_uids.append(uid)

    return np.array(avail_uids)


def get_random_uids(
    self, k: int, exclude: List[int] = None
) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = np.array(random.sample(available_uids, k))
    return uids
