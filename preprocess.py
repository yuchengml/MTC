import pickle
import random
from typing import Any, List, Union

import numpy as np
import pandas as pd
from scapy.all import sniff
from scapy.compat import raw
from scapy.layers.dns import DNS
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding, Packet
from scipy import sparse
from sklearn.model_selection import train_test_split
import glob

# Traffic type labels
PREFIX_TO_TRAFFIC_ID = {
    'chat': 0,
    'email': 1,
    'file_transfer': 2,
    'streaming': 3,
    'torrent': 4,
    'voip': 5,
    'vpn_chat': 6,
    'vpn_email': 7,
    'vpn_file_transfer': 8,
    'vpn_streaming': 9,
    'vpn_torrent': 10,
    'vpn_voip': 11
}

# Application labels
PREFIX_TO_APP_ID = {
    'aim_chat': 0,
    'email': 1,
    'facebook': 2,
    'ftps': 3,
    'icq': 4,
    'gmail': 5,
    'hangouts': 6,
    'netflix': 7,
    'scp': 8,
    'sftp': 9,
    'skype': 10,
    'spotify': 11,
    'torrent': 12,
    'tor': 13,
    'vimeo': 14,
    'voipbuster': 15,
    'youtube': 16
}

# Auxiliary task labels
AUX_ID = {
    'all_chat': 0,
    'all_email': 1,
    'all_file_transfer': 2,
    'all_streaming': 3,
    'all_torrent': 4,
    'all_voip': 5
}


def reduce_tcp(
        packet: Packet,
        n_bytes: int = 20
) -> Packet:
    """ Reduce the size of TCP header to 20 bytes.

    Args:
        packet: Scapy packet.
        n_bytes: Number of bytes to reserve.

    Returns:
        IP packet.
    """
    if TCP in packet:
        # Calculate the TCP header length
        tcp_header_length = packet[TCP].dataofs * 32 / 8

        # Check if the TCP header length is greater than 20 bytes
        if tcp_header_length > n_bytes:
            # Reduce the TCP header length to 20 bytes
            packet[TCP].dataofs = 5  # 5 * 4 = 20 bytes
            del packet[TCP].options  # Remove any TCP options beyond the 20 bytes

            # Recalculate the TCP checksum
            del packet[TCP].chksum
            del packet[IP].chksum
            packet = packet.__class__(bytes(packet))  # Recreate the packet to recalculate checksums

            # Display the modified packet
            # print("Modified Packet:")
            # print(packet.show())
    return packet


def pad_udp(packet: Packet):
    """ Pad the UDP header to 20 bytes with zero.

    Args:
        packet: Scapy packet.

    Returns:
        IP packet.
    """
    if UDP in packet:
        # Get layers after udp
        layer_after = packet[UDP].payload.copy()

        # Build a padding layer
        pad = Padding()
        pad.load = "\x00" * 12

        # Concat the origin payload with padding layer
        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

    return packet


def packet_to_sparse_array(
        packet: Packet,
        max_length: int = 1500
) -> sparse.csr_matrix:
    """ Normalize the byte string and convert to sparse matrix

    Args:
        packet: Scapy packet.
        max_length: Max packet length

    Returns:
        Sparse matrix.
    """
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)

    arr = sparse.csr_matrix(arr, dtype=np.float32)
    return arr


def filter_packet(pkt: Packet):
    """ Filter packet approach following MTC author.

    Args:
        pkt: Scapy packet.

    Returns:
        Scapy packet if pass all filtering rules. Or `None`.
    """
    # eliminate Ethernet header with the physical layer information
    if Ether in pkt:
        # print('Ethernet header in packet')
        pkt = pkt[Ether].payload
    else:
        # print('Ethernet header not in packet')
        pass

    # IP header was changed to 0.0.0.0
    if IP in pkt:
        # print('IP header in packet')
        pkt[IP].src = "0.0.0.0"
        pkt[IP].dst = "0.0.0.0"
        # print(pkt[IP].src, pkt[IP].dst, 'after modification')
    else:
        # print('IP header not in packet')
        return None

    if TCP in pkt:
        # print('TCP header in packet')
        # print(f'Len of TCP packet: {len(pkt[TCP])}, payload: {len(pkt[TCP].payload)}')
        pkt = reduce_tcp(pkt)
        # print(f'Len of TCP packet: {len(pkt[TCP])}, payload: {len(pkt[TCP].payload)} after reducing')
    elif UDP in pkt:
        # print('UDP header in packet')
        # print(f'Len of UDP packet: {len(pkt[UDP])}, payload: {len(pkt[UDP].payload)}')
        pkt = pad_udp(pkt)
        # print(f'Len of UDP packet: {len(pkt[UDP])}, payload: {len(pkt[UDP].payload)} after padding')
    else:
        return None

    # Pre-define TCP flags
    FIN = 0x01
    SYN = 0x02
    RST = 0x04
    PSH = 0x08
    ACK = 0x10
    URG = 0x20
    ECE = 0x40
    CWR = 0x80

    # Parsing transport layer protocols using Scapy
    # Checking if it is an IP packet
    if IP in pkt:
        # Obtaining data from the IP layer
        ip_packet = pkt[IP]

        # If it is a TCP protocol
        if TCP in ip_packet:
            # Obtaining data from the TCP layer
            tcp_packet = ip_packet[TCP]
            # Checking for ACK, SYN, FIN flags
            if tcp_packet.flags & 0x16 in [ACK, SYN, FIN]:
                # print('TCP has ACK, SYN, and FIN packets')
                # print(pkt)
                # Returning None (or an empty packet b'')
                return None
        # If it is a UDP protocol
        elif UDP in ip_packet:
            # Obtaining data from the UDP layer
            udp_packet = ip_packet[UDP]
            # Checking for DNS protocol (assuming the value is 53)
            if udp_packet.dport == 53 or udp_packet.sport == 53 or DNS in pkt:
                # print('UDP has DNS packets')
                # print(pkt)
                # Returning None (or an empty packet b'')
                return None
        else:
            # Not a TCP or UDP packet
            return None

        # Valid packet
        return pkt

    else:
        # Not an IP packet
        return None


def split_data(
        data: List[sparse.csr_matrix],
        train_ratio: float = 0.64,
        val_ratio: float = 0.16,
        test_ratio: float = 0.20,
        random_state: int = 42
) -> Union[Any, Any, Any]:
    """ Split data to training, validation and testing data.

    Args:
        data: List of features.
        train_ratio: The ratio of training data.
        val_ratio: The ratio of validation data.
        test_ratio: The ratio of testing data.
        random_state: Random seed.

    Returns:
        None.
    """
    # Check if the ratios add up to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios should add up to 1"

    # Splitting into temporary train and test sets
    train_val, test = train_test_split(data, test_size=test_ratio, random_state=random_state)

    # Calculate the ratios for train and validation sets based on the remaining data after test split
    remaining_ratio = 1 - test_ratio
    # train_ratio_adjusted = train_ratio / remaining_ratio
    val_ratio_adjusted = val_ratio / remaining_ratio

    # Split the remaining data into train and validation sets
    train, val = train_test_split(train_val, test_size=val_ratio_adjusted, random_state=random_state)

    return train, val, test


def preprocess_data(
        annotation_csv: str = 'data/annotation.csv',
        pcap_dir: str = 'pcaps',
        limited_count: int = None
):
    """ Perform data preprocessing

    Args:
        annotation_csv: Annotation file with CSV format which map the *.pcap* file names and labels of the three tasks.
        pcap_dir: The folder stored *.pcap* files.
        limited_count: Limited amount of record to fetch from *.pcap* files.

    Returns:

    """
    # Load annotation file
    annotation_df = pd.read_csv(annotation_csv)

    # Group applications to filter data for each class
    g = annotation_df.groupby('application_class')
    for group in g.groups:
        print(f'Filter `{group}` application class data')
        df = g.get_group(group)

        data_rows = []

        # Parse each *.pcap* file from each application class
        for pcap_f_name, traffic, app, aux in df.to_records(False).tolist():
            print(f'Load file: {pcap_f_name}')

            pkt_arrays = []

            # Callback function for sniffing
            def method_filter(pkt):
                # Eliminate Ethernet header with the physical layer information
                pkt = filter_packet(pkt)
                # A valid packet would be returned
                if pkt is not None:
                    # Convert to sparse matrix
                    ary = packet_to_sparse_array(pkt)
                    pkt_arrays.append(ary)

            # Limit the number of data for testing
            # Hint: `sniff` is a better way to parse *.pcap* files for saving memory, comparing to `rdpcap`
            if limited_count:
                sniff(offline=f'{pcap_dir}/{pcap_f_name}', prn=method_filter, store=0, count=10)
            else:
                sniff(offline=f'{pcap_dir}/{pcap_f_name}', prn=method_filter, store=0)

            # Concat feature and labels
            for array in pkt_arrays:
                row = {
                    "app_label": PREFIX_TO_APP_ID.get(app),
                    "traffic_label": PREFIX_TO_TRAFFIC_ID.get(traffic),
                    "aux_label": AUX_ID.get(aux),
                    "feature": array
                }
                data_rows.append(row)

            # Release memory
            del pkt_arrays

        print(f'Save `{group}` application class data with {len(data_rows)} rows')

        # Save a preprocessed data to pickle file
        with open(f'data/data_rows_{group}.pkl', 'wb') as f:
            pickle.dump(data_rows, f)

        # Release memory
        del data_rows


def collect_data(n: int = 50000):
    """Collect data from each application class with down-sampling"""
    sampled_data_rows = []

    # Load per class data from pickle files
    data_rows_files = glob.glob('data_rows_*')
    for file in data_rows_files:
        with open(file, 'rb') as f:
            data_rows = pickle.load(f)

        # Shuffle data
        for _ in range(10):
            random.shuffle(data_rows)

        data_rows = data_rows[:n]
        sampled_data_rows += data_rows

    # Shuffle data
    for _ in range(10):
        random.shuffle(sampled_data_rows)

    # Split data to training, validation and testing data
    train_data, val_data, test_data = split_data(sampled_data_rows)

    # Store preprocessed adn splited data to pickle files
    with open('data/train_data_rows.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('data/val_data_rows.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open('data/test_data_rows.pkl', 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    preprocess_data()
    collect_data()
