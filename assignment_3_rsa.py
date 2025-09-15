def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# Modular exponentiation
def mod_pow(base, exp, n):
    result = 1
    base = base % n
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % n
        exp = exp >> 1
        base = (base * base) % n
    return result

# Extended Euclidean Algorithm to find modular inverse
def mod_inverse(e, phi):
    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)
    
    g, x, _ = egcd(e, phi)
    if g != 1:
        raise Exception("No modular inverse")
    return x % phi

# RSA Key Generation (toy example)
p, q = 61, 53
n = p * q
phi = (p - 1) * (q - 1)
e = 17  # public exponent
d = mod_inverse(e, phi)

print(f"Public Key: (e={e}, n={n})")
print(f"Private Key: (d={d}, n={n})")

# Message input
msg = input("\nEnter an alphanumeric message: ")

# Encryption
encrypted = [mod_pow(ord(ch), e, n) for ch in msg]
print("\nEncrypted:", encrypted)

# Decryption
decrypted = ''.join(chr(mod_pow(c, d, n)) for c in encrypted)
print("Decrypted:", decrypted)