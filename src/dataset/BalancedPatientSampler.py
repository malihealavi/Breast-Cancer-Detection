import random 
import pickle
# -----------------------------------------
#             Balance DF
# -----------------------------------------
class BalancedPatientSampler:
    def __init__(self, neg_patient_ids, seed=42):
        self.seed = seed
        self.neg_ids = sorted(neg_patient_ids)
        self.rng = random.Random(seed)
        self.rng.shuffle(self.neg_ids)
        self.ptr = 0

    def sample(self, n_needed):
        out = []
        while n_needed > 0:
            remaining = len(self.neg_ids) - self.ptr
            take = min(remaining, n_needed)
            out.extend(self.neg_ids[self.ptr : self.ptr + take])
            self.ptr += take
            n_needed -= take
            if self.ptr >= len(self.neg_ids):
                self.rng.shuffle(self.neg_ids)
                self.ptr = 0
        return out

    def state_dict(self):
        return {
            'neg_ids': list(self.neg_ids),
            'rng_state': pickle.dumps(self.rng.getstate()), 
            'ptr': self.ptr,
        }
        
    def load_state_dict(self, state):
        try:
            self.neg_ids = list(state['neg_ids'])
            self.rng.setstate(pickle.loads(state['rng_state']))
            self.ptr = state['ptr']
        except Exception as e:
            print(f"[warn] neg_sampler state incompatible ({e}); reseeding fresh.")

