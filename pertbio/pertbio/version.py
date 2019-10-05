__version__ = '0.1.1'
VERSION = __version__

def get_msg():
    # for test: installation completed
    changelog = """
        version 0.0.2
        -- Jan 20, 2019 --
        * Huge Bug fixed with gradually increasing nT
        * Reorganize utils.py

        version 0.0.3
        -- Jan 21, 2019 --
        * Adding test of convergece

        version 0.0.3.1
        -- Jan 23, 2019 --
        * Roll back to x_0 = 1

        version 0.0.3
        -- Jan 21, 2019 --
        * Roll back 0.0.3

        version 0.0.3.2
        -- Jan 26, 2019 --
        * Adding outputs for test_convergence()

        version 0.0.3.3
        -- Jan 30, 2019 --
        * use last 20 time step for test_convergence()

        version 0.0.3.4
        -- Jan 31, 2019 --
        * use 0.1 as initial values for alpha and eps variable

        version 0.0.4
        -- Feb 11, 2019 --
        * Roll back 0.0.3.3
        * Add constraints on direct regulation from drug nodes to phenotypic nodes

        version 0.0.5
        -- Feb 21, 2019 --
        * Add function to normalize mse loss to different nodes.

        version 0.1.0
        -- Aug 21, 2019 --
        * Re-structure codes for publish.

        version 0.1.1
        -- Oct 4, 2019 --
        * Add new kinetics
        * Add new ODE solvers
        * Add new envelop forms
        """

    print(changelog)
