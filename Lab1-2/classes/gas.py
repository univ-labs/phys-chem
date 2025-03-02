from math import log


class Gas:
    def __init__(self, substance: str, volume: int, pressure: int, temperature: int):
        self._substance = substance
        self._volume = volume * 1e-4
        self._pressure = pressure
        self._temperature = temperature
        self.R = 8.314

    def __str__(self):
        return [self._substance, self._volume, self._temperature, self._pressure]

    def get_moles(self) -> float:
        return (self._pressure * self._volume) / (self.R * self._temperature)

    @property
    def get_volume(self):
        return self._volume


class GasMixing:

    def __init__(self, gas1: Gas, gas2: Gas):
        self._gas1 = gas1
        self._gas2 = gas2
        self._n1 = self._gas1.get_moles()
        self._n2 = self._gas2.get_moles()
        self._total_volume = self._gas1.get_moles() + self._gas2.get_moles()
        self.R = 8.314

    def get_mole_fraction(self, gas: Gas) -> float:
        return gas.get_moles() / self._total_volume

    def find_delta_entropy(self) -> float:
        return -(self._n1 + self._n2) * self.R * (
                    log(self._gas1.get_volume / self._total_volume) + log(self._gas2.get_volume / self._total_volume))
