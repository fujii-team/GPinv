from GPflow import gpmc
from .vgp import TransformedVGP
from .gpmc import TransformedGPMC
from .svgp import TransformedSVGP

class GPMC(TransformedGPMC):
    pass

class VGP(TransformedVGP):
    pass

class SVGP(TransformedSVGP):
    pass
