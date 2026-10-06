import numpy as np

import functionsSetup as fs
from generate_scenario import compute_interference


L_REALISTIC = 30                    ## small cells in the 1 km2 area
CARRIER_HZ = fs.B                   ## 20 MHz per antenna
CAPACITY_REALISTIC = 4              ## users (channels) per antenna
CHANNEL_HZ = CARRIER_HZ / CAPACITY_REALISTIC


class DynamicScenario:

    def __init__(self, L=L_REALISTIC, K=60, N=1, tau_p=4, ASD_varphi=10 * (np.pi / 180), numActiveAPs=5,
                 grid=True, semilla=2, p_mW=100.0, change_range=(0.2, 0.5), churn_share=0.5, max_move_m=90.0,
                 interference_scope="all", channel_Hz=CHANNEL_HZ):
        ## channel_Hz: bandwidth of one user's channel (5 MHz).
        ## functionsSetup computes the noise over its full B; a user transmitting
        ## p_mW in a narrower channel sees B / channel_Hz times less noise.
        (self.gain, _, self.active, _, self.pilot,
         ap_pos, ue_pos, self.shadow) = fs.generateSetup(
            L=L, K=K, N=N, tau_p=tau_p, ASD_varphi=ASD_varphi, numActiveAPs=numActiveAPs,
            grid=grid, semilla=semilla, return_state=True,
        )
        self.L, self.K, self.tau_p, self.numActiveAPs = L, K, tau_p, numActiveAPs
        self.ap_pos = np.asarray(ap_pos).ravel()
        self.ue_pos = np.asarray(ue_pos).ravel()
        self.p_mW = p_mW
        self.change_range = change_range
        self.churn_share = churn_share
        self.max_move_m = max_move_m
        self.interference_scope = interference_scope
        self.channel_Hz = float(fs.B if channel_Hz is None else channel_Hz)
        self.rng = np.random.default_rng(None if semilla is False else semilla + 1)

        self.uids = np.arange(K)
        self.next_uid = K
        self.t = 0
        self.arrived, self.moved = [], []

    ## Per UE updates.

    def _update_gains(self, k):
        """Gain column and active cluster of seat k from its position and shadowing."""
        d = np.sqrt(fs.distanceVertical ** 2 + np.abs(self.ap_pos - self.ue_pos[k]) ** 2)
        self.gain[:, k] = fs.constantTerm - fs.alpha * np.log10(d) + self.shadow[k] - fs.noiseVariancedBm
        self.active[k] = np.argsort(d)[:self.numActiveAPs]

    ## Included although not relevant for this project (we asume there's enough channel
    ## for pilot contamination to be out of the map)
    def _assign_pilot(self, k):
        """Same rule as functionsSetup: pilot with least received power at k's master AP."""
        master = np.argmax(self.gain[:, k])
        others = np.arange(self.K) != k
        power = [np.sum(fs.db2pow(self.gain[master, others & (self.pilot == p)])) for p in range(self.tau_p)]
        self.pilot[k] = int(np.argmin(power))

    def _reflect(self, x):
        S = fs.squarelength
        x = np.abs(x)
        return np.where(x > S, 2 * S - x, x)

    ## timeslots

    def step(self):
        """Advance one timeslot and return its snapshot."""
        frac = self.rng.uniform(*self.change_range)
        changed = self.rng.choice(self.K, size=int(round(frac * self.K)), replace=False)
        n_churn = int(round(self.churn_share * len(changed)))
        self.arrived = sorted(int(k) for k in changed[:n_churn])
        self.moved = sorted(int(k) for k in changed[n_churn:])

        for k in self.moved:
            d = self.rng.uniform(0, self.max_move_m)
            pos = self.ue_pos[k] + d * np.exp(1j * self.rng.uniform(0, 2 * np.pi))
            self.ue_pos[k] = self._reflect(pos.real) + 1j * self._reflect(pos.imag)
            a = 2 ** (-d / fs.decorr)
            self.shadow[k] = a * self.shadow[k] + np.sqrt(1 - a ** 2) * fs.sigma_sf * self.rng.standard_normal(self.L)
            self._update_gains(k)

        for k in self.arrived:
            self.ue_pos[k] = complex(*self.rng.uniform(0, fs.squarelength, 2))
            self.shadow[k] = fs.sigma_sf * self.rng.standard_normal(self.L)   ## new UE independent shadowing
            self.uids[k] = self.next_uid
            self.next_uid += 1
            self._update_gains(k)
        for k in self.arrived:   ## after all positions are known
            self._assign_pilot(k)

        self.t += 1
        return self.snapshot()

    def snapshot(self):
        """Inputs of the current timeslot, in the same form as get_data."""
        powgain = self.p_mW * (fs.B / self.channel_Hz) * fs.db2pow(self.gain)
        distances = np.sqrt(fs.distanceVertical ** 2 + np.abs(self.ap_pos[:, None] - self.ue_pos[None, :]) ** 2)
        return {
            "t": self.t,
            "distances": distances,          ## (L, K) AP-UE distance in m
            "ap_pos": self.ap_pos.copy(),
            "ue_pos": self.ue_pos.copy(),
            "active_APs": self.active.copy(),
            "gainOverNoisedB": self.gain.copy(),
            "powgain": powgain,
            "pilotIndex": self.pilot.copy(),
            "interference_matrix": compute_interference(powgain, self.active, self.interference_scope),
            "uids": self.uids.copy(),
            "channel_Hz": self.channel_Hz,   ## to turn bits/s/Hz into bits/s.
            "arrived": list(self.arrived),   ## seats with a new UE in this timeslot.
            "moved": list(self.moved),       ## seats whose UE moved into this timeslot.
        }

    def run(self, T):
        """Snapshots of T timeslots: the initial scenario, then T-1 steps."""
        return [self.snapshot()] + [self.step() for _ in range(T-1)]
