
import unittest

import pv


cfg_src = './config.yaml'

class TestTransmittanceModel(unittest.TestCase):
    """Test fit.transmittance_model with some (for now) nonsensical, made-up numbers.

    """
    def test_transmittance_model(self):

        # TODO: more realistic numbers:
        wavelengths = (200.,300.,400.,500.)
        alpha = 0.1
        theta = (3,2,10)
        epsilon = (2,3,4,5)
        # approximate results:
        verif_results = [98583.36997812, 200472.81869868, 322296.5813344, 455510.59075078]

        model_results = pv.utils.transmittance_model( wavelengths, alpha, *theta, epsilon=epsilon)

        self.assertEqual(len(verif_results),len(model_results))

        for i in range(len(verif_results)):
            self.assertAlmostEqual(verif_results[i],model_results[i],places=8)


if __name__=='__main__':
    unittest.main()

