"""
This module defines the version of the package
"""

__version__ = '0.3.1'
VERSION = __version__


def get_msg():
    """get version history"""
    # for test: installation completed
    changelog = [
        """
        version 0.0.2
        -- Jan 20, 2019 --
        * Huge Bug fixed with gradually increasing nT
        * Reorganize utils.py
        """,

        """
        version 0.0.3
        -- Jan 21, 2019 --
        * Adding test of convergece
        """,

        """
        version 0.0.3.1
        -- Jan 23, 2019 --
        * Roll back to x_0 = 1
        """,

        """
        version 0.0.3
        -- Jan 21, 2019 --
        * Roll back 0.0.3
        """,

        """
        version 0.0.3.2
        -- Jan 26, 2019 --
        * Adding outputs for test_convergence()
        """,

        """
        version 0.0.3.3
        -- Jan 30, 2019 --
        * use last 20 time step for test_convergence()
        """,

        """
        version 0.0.3.4
        -- Jan 31, 2019 --
        * use 0.1 as initial values for alpha and eps variable
        """,

        """
        version 0.0.4
        -- Feb 11, 2019 --
        * Roll back 0.0.3.3
        * Add constraints on direct regulation from drug nodes to phenotypic nodes
        """,

        """
        version 0.0.5
        -- Feb 21, 2019 --
        * Add function to normalize mse loss to different nodes.
        """,

        """
        version 0.1.0
        -- Aug 21, 2019 --
        * Re-structure codes for publish.
        """,

        """
        version 0.1.1
        -- Oct 4, 2019 --
        * Add new kinetics
        * Add new ODE solvers
        * Add new envelop forms
        """,

        """        
        version 0.2.0
        -- Feb 26, 2020 --
        * Add support of matrix operation rather than function mapping
        * Roughly 5x faster
        """,

        """        
        version 0.2.1
        -- Apr 5, 2020 --
        * Reformat for better code style
        * Revise docs
        """,

        """
        version 0.2.2
        -- Apr 23, 2020 --
        * Add support to tf.Datasets
        * Add support to tf.sparse
        * Prepare for sparse single-cell data
        """,

        """        
        version 0.2.3
        -- June 8, 2020 --
        * Add support to L2 loss (alone or together with L1, i.e. elastic net)
        * Clean the example configs folder
        """,

        """
        version 0.3.0
        -- June 8, 2020 --
        Add support for alternative form of perturbation
        * Previous: add u on activity nodes
        * New: fix activity nodes directly
            - 1) changing the x_0 from zeros to u
            - 2) adding mask on dxdt
            - 3) the previous format should work fine due to numpy broadcast
        * Revised printing log
        """,

        """
        version 0.3.1
        -- Sept 25, 2020 --
        * Release version for publication
        * Add documentation        
        """
    ]
    print(
        "Running CellBox scripts developed in Sander lab\n"
        "Maintained by Bo Yuan, Judy Shen, and Augustin Luna"
    )
    print(changelog[-1])
    print(
        "Tutorials and documentations are available at https://github.com/dfci/CellBox\n"
        "If you want to discuss the usage or to report a bug, please use the 'Issues' function at GitHub\n"
        "If you find CellBox useful for your research, please consider citing the corresponding publication.\n"
        "For more information, please email us at boyuan@g.harvard.edu and c_shen@g.harvard.edu, "
        "augustin_luna@hms.harvard.edu\n"
    )
