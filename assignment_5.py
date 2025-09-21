# Raw DES (pure Python implementation)
# Supports encryption/decryption of alphanumeric (and general ASCII) strings
# without using any external crypto library.
# WARNING: DES is outdated/insecure. This implementation is for educational use only.

# -- Tables from the DES standard --
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

FP = [40, 8, 48, 16, 56, 24, 64, 32,
      39, 7, 47, 15, 55, 23, 63, 31,
      38, 6, 46, 14, 54, 22, 62, 30,
      37, 5, 45, 13, 53, 21, 61, 29,
      36, 4, 44, 12, 52, 20, 60, 28,
      35, 3, 43, 11, 51, 19, 59, 27,
      34, 2, 42, 10, 50, 18, 58, 26,
      33, 1, 41, 9, 49, 17, 57, 25]

E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

P = [16, 7, 20, 21,
     29, 12, 28, 17,
     1, 15, 23, 26,
     5, 18, 31, 10,
     2, 8, 24, 14,
     32, 27, 3, 9,
     19, 13, 30, 6,
     22, 11, 4, 25]

SBOX = [
[[14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
 [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
 [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
 [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13]],

[[15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
 [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
 [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
 [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9]],

[[10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
 [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
 [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
 [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12]],

[[7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
 [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
 [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
 [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14]],

[[2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
 [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
 [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
 [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3]],

[[12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
 [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
 [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
 [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13]],

[[4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
 [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
 [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
 [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12]],

[[13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
 [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
 [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
 [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11]]
]

PC1 = [57,49,41,33,25,17,9,
       1,58,50,42,34,26,18,
       10,2,59,51,43,35,27,
       19,11,3,60,52,44,36,
       63,55,47,39,31,23,15,
       7,62,54,46,38,30,22,
       14,6,61,53,45,37,29,
       21,13,5,28,20,12,4]

PC2 = [14,17,11,24,1,5,
       3,28,15,6,21,10,
       23,19,12,4,26,8,
       16,7,27,20,13,2,
       41,52,31,37,47,55,
       30,40,51,45,33,48,
       44,49,39,56,34,53,
       46,42,50,36,29,32]

SHIFT_SCHEDULE = [1, 1, 2, 2, 2, 2, 2, 2,
                  1, 2, 2, 2, 2, 2, 2, 1]

# -- Helper bit/byte utilities --

def _to_bits(data_bytes):
    """Convert bytes to list of bits (MSB first)."""
    bits = []
    for b in data_bytes:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _from_bits(bits):
    """Convert list of bits (MSB first) to bytes."""
    b = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        b.append(byte)
    return bytes(b)


def _permute(bits, table):
    return [bits[i - 1] for i in table]


def _left_rotate(lst, n):
    return lst[n:] + lst[:n]


def _xor(a, b):
    return [x ^ y for x, y in zip(a, b)]

# -- Key schedule generation --

def generate_subkeys(key8bytes):
    if len(key8bytes) != 8:
        raise ValueError("Key must be 8 bytes long")
    key_bits = _to_bits(key8bytes)
    # Apply PC-1 (56 bits)
    permuted = _permute(key_bits, PC1)
    C = permuted[:28]
    D = permuted[28:]
    subkeys = []
    for shift in SHIFT_SCHEDULE:
        C = _left_rotate(C, shift)
        D = _left_rotate(D, shift)
        combined = C + D
        subkey = _permute(combined, PC2)  # 48 bits
        subkeys.append(subkey)
    return subkeys

# -- Feistel (F) function --

def feistel(R, subkey):
    # Expand R (32->48)
    ER = _permute(R, E)
    XR = _xor(ER, subkey)
    # S-box substitution (48 -> 32)
    out_bits = []
    for i in range(8):
        chunk = XR[i*6:(i+1)*6]
        row = (chunk[0] << 1) | chunk[5]
        col = (chunk[1] << 3) | (chunk[2] << 2) | (chunk[3] << 1) | chunk[4]
        val = SBOX[i][row][col]
        # 4 bits
        for k in range(3, -1, -1):
            out_bits.append((val >> k) & 1)
    # P permutation
    return _permute(out_bits, P)

# -- Encrypt/Decrypt single 64-bit block --

def encrypt_block(block8bytes, subkeys):
    bits = _to_bits(block8bytes)
    bits = _permute(bits, IP)
    L = bits[:32]
    R = bits[32:]
    for i in range(16):
        newL = R
        f_out = feistel(R, subkeys[i])
        newR = _xor(L, f_out)
        L, R = newL, newR
    preoutput = R + L  # note swap of L and R
    cipher_bits = _permute(preoutput, FP)
    return _from_bits(cipher_bits)


def decrypt_block(block8bytes, subkeys):
    bits = _to_bits(block8bytes)
    bits = _permute(bits, IP)
    L = bits[:32]
    R = bits[32:]
    for i in range(15, -1, -1):
        newL = R
        f_out = feistel(R, subkeys[i])
        newR = _xor(L, f_out)
        L, R = newL, newR
    preoutput = R + L
    plain_bits = _permute(preoutput, FP)
    return _from_bits(plain_bits)

# -- Padding (PKCS#5/7 style for 8-byte blocks) --

def pad(data_bytes):
    pad_len = 8 - (len(data_bytes) % 8)
    if pad_len == 0:
        pad_len = 8
    return data_bytes + bytes([pad_len] * pad_len)


def unpad(padded):
    if len(padded) == 0 or len(padded) % 8 != 0:
        raise ValueError("Invalid padded data length")
    pad_len = padded[-1]
    if pad_len < 1 or pad_len > 8:
        raise ValueError("Invalid padding")
    if padded[-pad_len:] != bytes([pad_len] * pad_len):
        raise ValueError("Invalid padding bytes")
    return padded[:-pad_len]

# -- High level ECB mode (simple) --

def des_encrypt_ecb(plaintext: str, key: str) -> bytes:
    """Encrypt plaintext string (ASCII/UTF-8) using 8-char key. Returns raw bytes ciphertext."""
    key_bytes = key.encode('utf-8')
    subkeys = generate_subkeys(key_bytes)
    data = plaintext.encode('utf-8')
    data_padded = pad(data)
    ciphertext = bytearray()
    for i in range(0, len(data_padded), 8):
        block = data_padded[i:i+8]
        enc = encrypt_block(block, subkeys)
        ciphertext.extend(enc)
    return bytes(ciphertext)


def des_decrypt_ecb(ciphertext: bytes, key: str) -> str:
    key_bytes = key.encode('utf-8')
    subkeys = generate_subkeys(key_bytes)
    plain_padded = bytearray()
    for i in range(0, len(ciphertext), 8):
        block = ciphertext[i:i+8]
        dec = decrypt_block(block, subkeys)
        plain_padded.extend(dec)
    plain = unpad(bytes(plain_padded))
    return plain.decode('utf-8', errors='strict')

# -- Utility to show hex and parse hex --

def bytes_to_hex(b: bytes) -> str:
    return ''.join(f"{x:02x}" for x in b)


def hex_to_bytes(h: str) -> bytes:
    h = h.strip()
    if len(h) % 2:
        raise ValueError("Invalid hex length")
    return bytes(int(h[i:i+2], 16) for i in range(0, len(h), 2))

# -- Example usage if run as script --
if __name__ == '__main__':
    print("Note: key must be exactly 8 characters (8 bytes).")

    key = input("Enter 8-character key: ")
    if len(key.encode('utf-8')) != 8:
        print("Error: key must be exactly 8 bytes long when encoded in UTF-8.")
        raise SystemExit(1)

    choice = input("(E)ncrypt or (D)ecrypt? ").strip().upper()
    if choice == 'E':
        plaintext = input("Enter plaintext (alphanumeric): ")
        ct = des_encrypt_ecb(plaintext, key)
        print("Ciphertext (hex):", bytes_to_hex(ct))
    elif choice == 'D':
        h = input("Enter ciphertext (hex): ")
        ct_bytes = hex_to_bytes(h)
        pt = des_decrypt_ecb(ct_bytes, key)
        print("Plaintext:", pt)
    else:
        print("Invalid choice")
