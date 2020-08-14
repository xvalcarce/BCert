# coding: utf-8

import numpy as np
from itertools import product, combinations

class CompactSpace(object):
    """ Compact space class.

    """
    def __init__(self, bound):
        self.__bound = self.__check_dim_bound(bound)
        self.__dim = len(bound)

    def __check_dim_bound(self,bound):
        """ Safety check on the format of provided bound.

        Parameters
        ---------
        bound : numpy.ndarray
            2 dimension list

        Returns
        -------
        bound : numpy.ndarray
            Return the input `bound` if the format is correct. Raise an exception otherwise.
        """
        if len(bound.shape) != 2:
            raise ValueError("bound should be of two dimensions, found dimension {}".format(len(bound.shape)))
        elif len(bound[0]) !=2:
            raise ValueError("bound should contain exactly two elements per dimension")
        else:
            return bound

    def discretized(self, step, orientation=None):
        """ Discretized the domain (cf. `bound`) into subdomain -- hypercubes centered on points from a grid.

        Parameters
        ----------
        step : int or list
            Seperation between grid points.

        Returns
        -------
        grid : numpy.ndarray
            Discretized space i.e. grid.
        """
        if type(step) in [int,float]:
            return self.discretized_symmetric(step,orientation)
        elif type(step) is np.ndarray:
            return self.discretized_asymmetric(step)

    def discretized_symmetric(self, step, orientation=None):
        """ Discretized the domain. Grid with same distance between first nieghboring points.

        Parameters
        ----------
        step : int
            Seperation between grid points.

        Returns
        -------
        grid : numpy.ndarray
            Discretized space i.e. grid with distance `step` between first nieghbors.
        """
        X = np.meshgrid(*[np.arange(b[0]+step/2,b[1],step) for b in self.__bound])
        grid = np.array([x.ravel() for x in X]).T
        if orientation is not None:
            add_grid = self.orientation_grid(grid, orientation, step)
            grid = np.concatenate((grid,add_grid))
        return grid

    def discretized_asymmetric(self, step):
        """ Discretized the domain. Grid with same distance between first nieghboring points.

        Parameters
        ----------
        step : list
            Seperation between grid points. `n`th element is the distance between point in the `n` direction.

        Returns
        -------
        grid : numpy.ndarray
            Discretized space i.e. grid.
        """
        raise NotImplemented("Function not implemented -- not compatible with discr_elem currently.")

    def discr_element(self, elem, step):
        """ Create new grid from a single point.

        Parameters
        ----------
        elem : list or numpy.array
            One dimensional list repsenting a point in the domain of definition.
        step : int
            Step, the distance between first neighbors grid point.

        Returns
        -------
        grid : numpy.ndarray
            A new grid, each element are the center `elem` shifted in all diagonal of the space.
        """
        if len(elem) != self.__dim:
            raise ValueError("Provided element is not in the same space as the original compact space")
        grid = []
        new_center = step*np.array([c for c in product(range(-1,2,2), repeat=self.__dim)])
        for c in new_center:
            if (np.all([c[i]+e > self.__bound[i][0] for i,e in enumerate(elem)]) and
                np.all([c[i]+e < self.__bound[i][1] for i,e in enumerate(elem)])):
                    grid.append(c+elem)
        return np.array(grid)

    def orientation_grid(self, grid, orientation, step):
        s = step/2
        ind = np.nonzero(orientation)[0]
        c_ind = [_ for i in range(ind.size) for _ in list(combinations(ind,i+1))]
        no = [[_ if not i in c else _*-1 for i,_ in enumerate(orientation)] for c in c_ind]
        add_grid = []
        for g in grid:
            for o in no:
                add_grid.append([_+(s*o[i]) for i,_ in enumerate(g)])
        return np.array(add_grid)
