import numpy as np
from numpy.polynomial import polynomial as poly

class CKKSImplementation:
    def __init__(self, N, Q, scale):
        """
        N: Degree of the polynomial (must be a power of 2)
        Q: Ciphertext modulus
        scale: The scaling factor (Delta) for fixed-point arithmetic
        """
        self.N = N
        self.Q = Q
        self.scale = scale
        # Polynomial modulus x^N + 1
        self.poly_mod = np.zeros(N + 1)
        self.poly_mod[0] = 1
        self.poly_mod[N] = 1

    def _poly_add(self, p1, p2):
        return poly.polyadd(p1, p2) % self.Q

    def _poly_mul(self, p1, p2):
        # Polynomial multiplication followed by reduction modulo x^N + 1
        res = poly.polymul(p1, p2)
        _, remainder = poly.polydiv(res, self.poly_mod)
        return np.round(remainder) % self.Q

    def key_gen(self):
        # Step 1: Key Generation
        # Secret Key sk: coefficients from {-1, 0, 1}
        self.sk = np.random.randint(-1, 2, self.N)
        # Random polynomial a
        self.a = np.random.randint(0, self.Q, self.N)
        # Noise e
        self.e = np.random.normal(0, 2, self.N).astype(int)
        
        # Public Key pk = (b, a) where b = -a*sk + e
        b = self._poly_add(-self._poly_mul(self.a, self.sk), self.e)
        self.pk = (b, self.a)
        print("Keys generated successfully.")

    def encrypt(self, m_vec):
        # Step 2: Encoding and Encryption
        # (Simplified) Encoding: scale and treat as polynomial coefficients
        m_enc = (np.array(m_vec) * self.scale).astype(int)
        
        v = np.random.randint(-1, 2, self.N)
        e0 = np.random.normal(0, 2, self.N).astype(int)
        e1 = np.random.normal(0, 2, self.N).astype(int)
        
        # c0 = m_enc + v*pk[0] + e0
        c0 = self._poly_add(self._poly_add(m_enc, self._poly_mul(v, self.pk[0])), e0)
        # c1 = v*pk[1] + e1
        c1 = self._poly_add(self._poly_mul(v, self.pk[1]), e1)
        
        return (c0, c1)

    def decrypt(self, ct):
        # Step 4: Decryption
        # m = c0 + c1 * sk
        c0, c1 = ct
        m_approx = self._poly_add(c0, self._poly_mul(c1, self.sk))
        
        # Decoding and rescaling
        # Handle modular wrap-around for negative numbers
        m_decoded = m_approx.copy()
        m_decoded[m_decoded > self.Q // 2] -= self.Q
        
        return m_decoded / self.scale

    def add(self, ct1, ct2):
        # Step 3: Homomorphic Addition
        return (self._poly_add(ct1[0], ct2[0]), self._poly_add(ct1[1], ct2[1]))

# --- Execution Example ---
if __name__ == "__main__":
    # Parameters
    N = 8  # Polynomial degree
    Q = 10**8  # Large modulus
    scale = 10**3 # Scale factor
    
    ckks = CKKSImplementation(N, Q, scale)
    ckks.key_gen()
    
    # Input vectors (length N)
    m1 = [1, 2, 3, 0, 0, 0, 0, 0]
    m2 = [4, 5, 6, 0, 0, 0, 0, 0]
    
    # Encrypt
    ct1 = ckks.encrypt(m1)
    ct2 = ckks.encrypt(m2)
    
    # Homomorphic Addition
    ct_sum = ckks.add(ct1, ct2)
    
    # Decrypt
    result = ckks.decrypt(ct_sum)
    
    print(f"Original m1 + m2: {np.array(m1) + np.array(m2)}")
    print(f"Decrypted Result: {result.round(1)}")