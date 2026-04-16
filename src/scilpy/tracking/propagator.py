# -*- coding: utf-8 -*-
from enum import Enum
import logging

import numpy as np

from dipy.data import get_sphere
from dipy.io.stateful_tractogram import Space, Origin
from dipy.reconst.shm import sh_to_sf_matrix, sph_harm_ind_list
from scipy.special import eval_legendre
from scipy import ndimage

from scilpy.reconst.utils import (get_sphere_neighbours,
                                  get_sh_order_and_fullness)
from scilpy.tracking.utils import sample_distribution, TrackingDirection
from scilpy.image.volume_space_management import FibertubeDataVolume


class PropagationStatus(Enum):
    ERROR = 1


class AbstractPropagator(object):
    """
    Abstract class for propagator object. "Propagation" means continuing the
    streamline a step further. The propagator is thus responsible for sampling
    the next direction at current step through Runge-Kutta integration
    (whereas the tracker using this propagator will be responsible for the
    processing parameters, number of streamlines, stopping criteria, etc.).

    Propagation depends on the type of data (ex, DTI, fODF) and the way to get
    a direction from it (ex, det, prob).
    """
    def __init__(self, datavolume, step_size, rk_order, space, origin):
        """
        Parameters
        ----------
        datavolume: scilpy.image.volume_space_management.DataVolume
            Trackable Dataset object.
        step_size: float
            The step size for tracking. Important: step size should be in the
            same units as the space of the tracking!
        rk_order: int
            Order for the Runge Kutta integration.
        space: dipy Space
            Space of the streamlines during tracking.
            value.
        origin: dipy Origin
            Origin of the streamlines during tracking. All coordinates received
            in the propagator's methods will be expected to respect
            that origin.

        A note on space and origin: All coordinates received in the
        propagator's methods will be expected to respect those values.
        Tracker will verify that the propagator has the same internal values as
        itself.
        """
        self.datavolume = datavolume

        self.origin = origin
        self.space = space

        # Propagation options
        self.step_size = step_size
        if not (rk_order == 1 or rk_order == 2 or rk_order == 4):
            raise ValueError("Invalid runge-kutta order. Is " +
                             str(rk_order) + ". Choices : 1, 2, 4")
        self.rk_order = rk_order

        # By default, normalizing directions. Adding option for child classes.
        self.normalize_directions = True

        # Will be reset at each new streamline.
        self.line_rng_generator = None

    def reset_data(self, new_data=None):
        """
        Reset data before starting a new process. In current implementation,
        we reset the internal data to None before starting a multiprocess, then
        load it back when process has started.

        Parameters
        ----------
        new_data: Any
            Will replace self.datavolume.data.

        """
        self.datavolume.data = new_data

    def prepare_forward(self, seeding_pos, random_generator):
        """
        Prepare information necessary at the first point of the
        streamline for forward propagation: v_in and any other information
        necessary for the self.propagate method.

        Parameters
        ----------
        seeding_pos: tuple(x,y,z)
            The seeding position. Important, position must be in the same space
            and origin as self.space, self.origin!
        random_generator: numpy Generator.

        Returns
        -------
        tracking_info: Any
            Any tracking information necessary for the propagation.
            Return PropagationStatus.ERROR if no good tracking direction can be
            set at current seeding position.
        """
        # To be defined by child classes.
        # Should set self.line_rng_generator = random_generator
        raise NotImplementedError

    def prepare_backward(self, line, forward_dir):
        """
        Called at the beginning of backward tracking, in case we need to
        reset some parameters

        Parameters
        ----------
        line: List
            Result from the forward tracking, reversed.
        forward_dir: ndarray (3,)
            v_in chosen at the forward step.

        Returns
        -------
        v_in: ndarray (3,)
            Last direction of the streamline. If the streamline contains
            only the seeding point (forward tracking failed), simply inverse
            the forward direction.
        """
        if len(line) > 1:
            v = line[-1] - line[-2]
            if self.normalize_directions:
                return v / np.linalg.norm(v)
            else:
                return v
        elif forward_dir is not None:
            return [-dir_i for dir_i in forward_dir]
        else:
            return None

    def finalize_streamline(self, last_pos, v_in):
        """
        Return the last position of the streamline.

        Parameters
        ----------
        last_pos: ndarray (3,)
            Last propagated position. Important, position must be in the same
            space and origin as self.space, self.origin!
        v_in: TrackingDirection
            Last propagated direction.

        Returns
        -------
        final_pos: ndarray (3,)
            Position of the final point of the streamline. Return None, or
            last_pos, if no last step is wished.
        """
        # Make a last step straight in the last direction (no sampling or
        # interpolation of a new direction). Ex of use: if stopped because it
        # exited the (WM) tracking mask, reaching GM a little more.
        final_pos = last_pos + self.step_size * np.array(v_in)
        return final_pos

    def _sample_next_direction_or_go_straight(self, pos, v_in):
        """
        Same as _sample_next_direction but if no valid direction has been
        found, return v_in as v_out.
        """
        is_direction_valid = True
        v_out = self._sample_next_direction(pos, v_in)
        if v_out is None:
            is_direction_valid = False
            v_out = v_in

        return is_direction_valid, v_out

    def propagate(self, line, v_in):
        """
        Given the current position and direction, computes the next position
        and direction using Runge-Kutta integration method. If no valid
        tracking direction is available, v_in is chosen.

        Parameters
        ----------
        line: list[ndarrray (3,)]
            Current position.
        v_in: ndarray (3,) or TrackingDirection
            Previous tracking direction.

        Return
        ------
        new_pos: ndarray (3,)
            The new segment position, expressed in propagator's space and
            origin.
        new_dir: ndarray (3,) or TrackingDirection
            The new segment direction.
        is_direction_valid: bool
            True if new_dir is valid.
        """
        # Finding last coordinate
        pos = line[-1]

        if self.rk_order == 1:
            is_direction_valid, new_dir = \
                self._sample_next_direction_or_go_straight(pos, v_in)

        elif self.rk_order == 2:
            is_direction_valid, dir1 = \
                self._sample_next_direction_or_go_straight(pos, v_in)
            _, new_dir = self._sample_next_direction_or_go_straight(
                pos + 0.5 * self.step_size * np.array(dir1), dir1)

        else:
            # case self.rk_order == 4
            is_direction_valid, dir1 = \
                self._sample_next_direction_or_go_straight(pos, v_in)
            v1 = np.array(dir1)
            _, dir2 = self._sample_next_direction_or_go_straight(
                pos + 0.5 * self.step_size * v1, dir1)
            v2 = np.array(dir2)
            _, dir3 = self._sample_next_direction_or_go_straight(
                pos + 0.5 * self.step_size * v2, dir2)
            v3 = np.array(dir3)
            _, dir4 = self._sample_next_direction_or_go_straight(
                pos + self.step_size * v3, dir3)
            v4 = np.array(dir4)

            new_v = (v1 + 2 * v2 + 2 * v3 + v4) / 6
            new_dir = TrackingDirection(new_v, dir1.index)

        new_pos = pos + self.step_size * np.array(new_dir)

        return new_pos, new_dir, is_direction_valid

    def _sample_next_direction(self, pos, v_in):
        """
        Chooses a next tracking direction from all possible directions offered
        by the tracking field.

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.  Important, position must be in the same
            space and origin as self.space, self.origin!
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        -------
        direction: ndarray (3,)
            A valid tracking direction. None if no valid direction is found.
            Direction should be normalized.
        """
        raise NotImplementedError


class ODFPropagator(AbstractPropagator):
    """
    Propagator on ODFs/fODFs. Algo can be det or prob.
    """
    def __init__(self, datavolume, step_size,
                 rk_order, algo, basis, sf_threshold, sf_threshold_init,
                 theta, dipy_sphere='symmetric724',
                 sub_sphere=0,
                 min_separation_angle=np.pi / 16.,
                 space=Space('vox'), origin=Origin('center'),
                 is_legacy=True):
        """

        Parameters
        ----------
        datavolume: scilpy.image.volume_space_management.DataVolume
            Trackable DataVolume object.
        step_size: float
            The step size for tracking.
        rk_order: int
            Order for the Runge Kutta integration.
        algo: string
            Type of algorithm. Choices are 'det' or 'prob'
        basis: string
            SH basis name. One of 'tournier07' or 'descoteaux07'
        sf_threshold: float
            Threshold on spherical function (SF).
        sf_threshold_init: float
            Threshold on spherical function when initializing a new streamline.
        theta: float
            Maximum angle (radians) between two steps.
        dipy_sphere: string, optional
            Name of the DIPY sphere object to use for evaluating SH. Can't be
            None.
        sub_sphere: int
            Number of subdivisions to use for the sphere.
        min_separation_angle: float, optional
            Minimum separation angle (in radians) for peaks extraction. Used
            for deterministic tracking. A candidate direction is a maximum if
            its SF value is greater than all other SF values in its
            neighbourhood, where the neighbourhood includes all the sphere
            directions located at most `min_separation_angle` from the
            candidate direction.
        space: dipy Space
            Space of the streamlines during tracking. Default: VOX, like in
            dipy. Interpolation of the ODF is done in VOX space (see
            DataVolume.vox_to_value) so this choice implies the less data
            modification.
        origin: dipy Origin
            Origin of the streamlines during tracking. Default: center, like in
            dipy. Interpolation of the ODF is done in center origin so this
            choice implies the less data modification.
        is_legacy : bool, optional
            Whether or not the SH basis is in its legacy form.
        """
        super().__init__(datavolume, step_size, rk_order, space, origin)

        self.iteration = 0
        self.sphere = get_sphere(name=dipy_sphere).subdivide(n=sub_sphere)
        self.dirs = np.zeros(len(self.sphere.vertices), dtype=np.ndarray)
        for i in range(len(self.sphere.vertices)):
            self.dirs[i] = TrackingDirection(self.sphere.vertices[i], i)

        if self.space == Space.RASMM:
            raise NotImplementedError(
                "This version of the propagator on ODF is not ready to work "
                "in RASMM space.")

        # Warn user if the rk order does not match the algo
        if rk_order != 1 and algo == 'prob':
            logging.warning('Probabilistic tracking with RK order != 1 is '
                            'not recommended! Use deterministic tracking '
                            'or set rk_order to 1 instead.')

        # Propagation params
        self.theta = theta
        if algo not in ['det', 'prob']:
            raise ValueError("ODFPropagator algo should be 'det' or 'prob'.")
        self.algo = algo
        self.tracking_neighbours = get_sphere_neighbours(self.sphere,
                                                         self.theta)
        # For deterministic tracking:
        self.maxima_neighbours = get_sphere_neighbours(self.sphere,
                                                       min_separation_angle)

        # ODF params
        self.sf_threshold = sf_threshold
        self.sf_threshold_init = sf_threshold_init
        sh_order, full_basis =\
            get_sh_order_and_fullness(self.datavolume.nb_coeffs)
        self.basis = basis
        self.is_legacy = is_legacy
        self.B = sh_to_sf_matrix(self.sphere, sh_order, self.basis,
                                 smooth=0.006, return_inv=False,
                                 full_basis=full_basis, legacy=self.is_legacy)

    def _get_sf(self, pos):
        """
        Get the spherical function at position pos.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in the trackable dataset. Important, position should be
            in the same space and origin as self.space, self.origin!

        Return
        ------
        sf: ndarray (len(self.sphere.vertices),)
            Spherical function evaluated at pos, normalized by
            its maximum amplitude.
        """
        # Interpolation:
        sh = self.datavolume.get_value_at_coordinate(
            *pos, space=self.space, origin=self.origin)
        sf = np.dot(self.B.T, sh).reshape((-1, 1))

        sf_max = np.max(sf)
        if sf_max > 0:
            sf /= sf_max
        return sf

    def prepare_backward(self, line, forward_dir):
        """
        Called at the beginning of backward tracking, in case we need to
        reset some parameters

        Parameters
        ----------
        line: List
            Result from the forward tracking, reversed.
        forward_dir: ndarray (3,)
            v_in chosen at the forward step.

        Returns
        -------
        v_in: ndarray (3,)
            Last direction of the streamline, of if it contains only the
            seeding point (forward tracking failed), simply inverse the
            forward direction.
        """
        if len(line) > 1:
            last_dir = line[-1] - line[-2]
            ind = self.sphere.find_closest(last_dir)
        else:
            backward_dir = -np.asarray(forward_dir)
            ind = self.sphere.find_closest(backward_dir)

        # toDo. Is using a TrackingDirection necessary compared to a direction
        #  x,y, z or rho, phi? self.sphere.vertices[ind] might not be
        #  exactly equal to last_dir or to backward_dir.
        return TrackingDirection(self.sphere.vertices[ind], ind)

    def prepare_forward(self, seeding_pos, random_generator):
        """
        Prepare information necessary at the first point of the
        streamline for forward propagation: v_in and any other information
        necessary for the self.propagate method.

        About **v_in**, it is used for two things:

        - To sample the next direction based on _sample_next_direction method.
            Ex, with fODF, it defines a cone theta of accepable directions.
        - If no valid next dir are found, continue straight.

        Parameters
        ----------
        seeding_pos: tuple(x,y,z)
            The seeding position. Important, position must be in the same space
            and origin as self.space, self.origin!
        random_generator: numpy Generator

        Returns
        -------
        v_in: TrackingDirection
            The "fake" previous direction at first step. Could be None if your
            propagator can propagate without knowledge of previous direction.
            Return PropagationStatus.Error if no good tracking direction can be
            set at current seeding position.
        """
        # Sampling on the SF values (no matter if general algo is det or prob)
        # with a different threshold than usual (sf_threshold_init).
        # So the initial step's propagation will be in a cone theta around a
        # "more probable" peak.
        self.iteration = 0
        sf = self._get_sf(seeding_pos)
        sf[sf < self.sf_threshold_init] = 0
        self.line_rng_generator = random_generator

        if np.sum(sf) > 0:
            ind = sample_distribution(sf, self.line_rng_generator)
            return TrackingDirection(self.dirs[ind], ind)

        # Else: sf at current position is smaller than acceptable threshold in
        # all directions.
        return PropagationStatus.ERROR

    def _sample_next_direction(self, pos, v_in):
        """
        Chooses a next tracking direction from all possible directions offered
        by the tracking field.

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.  Important, position must be in the same
            space and origin as self.space, self.origin!
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        -------
        direction: ndarray (3,)
            A valid tracking direction. None if no valid direction is found.
        """
        if self.algo == 'prob':
            # Tracking field returns the sf and directions
            sf, directions = self._get_possible_next_dirs_prob(pos, v_in)

            # Sampling one.
            if np.sum(sf) > 0:
                v_out = directions[
                    sample_distribution(sf, self.line_rng_generator)]
            else:
                return None
        elif self.algo == 'det':
            # Tracking field returns the list of possible maxima.
            possible_maxima = self._get_possible_next_dirs_det(pos, v_in)
            # Choosing one.
            cosinus = 0
            v_out = None
            for d in possible_maxima:
                new_cosinus = np.dot(v_in, d)
                if new_cosinus > cosinus:
                    cosinus = new_cosinus
                    v_out = d
        else:
            raise ValueError("Tracking choice must be one of 'det' or 'prob'.")

        # Not normalizing: direction comes from dipy's (unit) sphere so
        # supposing that it's ok.
        return v_out

    def _get_possible_next_dirs_prob(self, pos, v_in):
        """
        Get the spherical functions thresholded at position pos, for a given
        direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset. Important, position must be in the
            same space and origin as self.space, self.origin!
        v_in: TrackingDirection
            Incoming direction. Outcoming direction won't be further than an
            angle theta.

        Return
        ------
        value: tuple
            The neighbours SF evaluated at pos in given direction and
            corresponding tracking directions.
        """
        sf = self._get_sf(pos)
        sf[sf < self.sf_threshold] = 0
        inds = np.nonzero(
            self.tracking_neighbours[v_in.index])[0]
        return sf[inds], self.dirs[inds]

    def _get_possible_next_dirs_det(self, pos, previous_direction):
        """
        Get the set of maxima directions from the thresholded
        SF at position pos, for a direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset. Important, position must be in the
            same space and origin as self.space, self.origin!
        previous_direction: TrackingDirection
            Incoming direction. Outcoming direction won't be further than an
            angle theta.

        Return
        ------
        maxima: list
            List of directions of maxima around the input direction at pos.
        """
        sf = self._get_sf(pos)
        odf = sf.copy()
        sf[sf < self.sf_threshold] = 0
        maxima = []
        for i in np.nonzero(self.tracking_neighbours[previous_direction.index])[0]:
            if 0 < sf[i] == np.max(sf[self.maxima_neighbours[i]]):
                maxima.append(self.dirs[i])
        from fury import window, actor
        s = window.Scene()
        a = actor.odf_slicer(odf.reshape((1, 1, 1, -1)), sphere=self.sphere)
        if len(maxima) > 0:
            p = actor.peak_slicer(np.asarray(maxima))
            s.add(p)
        prevp = actor.peak_slicer(previous_direction, colors=(0, 1, 0))
        s.add(a, prevp)
        window.record(scene=s, out_path=f'tracking_it{self.iteration}.png')
        return maxima


class MicroscopyODFPropagator(ODFPropagator):
    """
    Docstring for MicroscopyPropagator
    """
    def __init__(self, datavolume, step_size, rk_order, window_halfwidth,
                 algo, sh_order_max, sf_threshold, sf_threshold_init,
                 theta, dipy_sphere='symmetric724', sub_sphere=0,
                 min_separation_angle=np.pi / 16.,
                 space=Space('vox'), origin=Origin('center')):
        # skip __init__ of ODF propagator as it expects SH coefficients as input
        AbstractPropagator.__init__(self, datavolume, step_size,
                                    rk_order, space, origin)

        self.iteration = 0
        self.sphere = get_sphere(name=dipy_sphere).subdivide(n=sub_sphere)
        self.dirs = np.zeros(len(self.sphere.vertices), dtype=np.ndarray)
        for i in range(len(self.sphere.vertices)):
            self.dirs[i] = TrackingDirection(self.sphere.vertices[i], i)

        # initialization of ODFPropagator class members
        if self.space != Space.VOX:
            raise NotImplementedError(
                "This version of the propagator is not ready to work "
                "in VOXMM/RASMM space.")

        # Warn user if the rk order does not match the algo
        if rk_order != 1 and algo == 'prob':
            logging.warning('Probabilistic tracking with RK order != 1 is '
                            'not recommended! Use deterministic tracking '
                            'or set rk_order to 1 instead.')

        # Propagation params
        self.theta = theta
        if algo not in ['det', 'prob']:
            raise ValueError("MicroscopyODFPropagator algo should be 'det' or 'prob'.")
        self.algo = algo
        self.tracking_neighbours = get_sphere_neighbours(self.sphere, self.theta)
        # For deterministic tracking:
        self.maxima_neighbours = get_sphere_neighbours(self.sphere, min_separation_angle)

        # ODF params
        self.sf_threshold = sf_threshold
        self.sf_threshold_init = sf_threshold_init
        self.sh_order = sh_order_max
        self.is_legacy = True
        self.B, self.B_inv = sh_to_sf_matrix(self.sphere, sh_order_max=sh_order_max,
                                             smooth=0.0, return_inv=True)

        self.iteration = 0
        self.derivative_order = 4

        # Parameters for estimating ODF from microscopy image
        self.halfwidth = window_halfwidth
        sampling_rate = 3.0 / (window_halfwidth * np.sqrt(2))
        self.wfilter_x, self.wfilter_y, self.wfilter_z = \
            np.meshgrid(*[np.arange(-self.halfwidth, self.halfwidth+1)*sampling_rate for _ in range(3)], indexing='ij')
        self.neighbour_offsets = np.stack(
            np.meshgrid(*[np.arange(-self.halfwidth, self.halfwidth + 1) for _ in range(3)],
                        indexing='ij'), axis=-1)

        # FRT definition
        self.n_coeffs = int((sh_order_max + 2) * (sh_order_max + 1)) / 2
        _, l = sph_harm_ind_list(sh_order_max)
        self.FRT = np.diag(2.0*np.pi*eval_legendre(l, 0))

        if self.derivative_order == 2:
            # G2/H2 filter pair normalization
            self.normalization_G2 = 2.0 / np.sqrt(3.0) * (2.0 / np.pi)**0.75
            self.normalization_H2 = 0.877776
            G2_basis_pairs = [
                self._G2a(), self._G2b(), self._G2c(),
                self._G2d(), self._G2e(), self._G2f()
            ]
            self.G2_filter = []
            self.G2_kappas = []
            for kappa, g2_filter in G2_basis_pairs:
                self.G2_kappas.append(kappa)
                self.G2_filter.append(g2_filter)
            self.G2_filter = np.stack(self.G2_filter, axis=-1)
            self.G2_kappas = np.asarray(self.G2_kappas)

            H2_basis_pairs = [
                self._H2a(), self._H2b(), self._H2c(), self._H2d(), self._H2e(), 
                self._H2f(), self._H2g(), self._H2h(), self._H2i(), self._H2j()
            ]
            self.H2_filter = []
            self.H2_kappas = []
            for kappa, h2_filter in H2_basis_pairs:
                self.H2_kappas.append(kappa)
                self.H2_filter.append(h2_filter)
            self.H2_filter = np.stack(self.H2_filter, axis=-1)
            self.H2_kappas = np.asarray(self.H2_kappas)
        elif self.derivative_order == 4:
            # G4 and H4 filters
            self.G4_filters = self._G4()
            self.H4_filters = self._H4()
        else:
            raise ValueError(f"derivative_order must be 2 or 4, got {self.derivative_order}")

    def _G4(self):
        x_prime = np.reshape(self.sphere.vertices[:, 0], (-1, 1, 1, 1)) * self.wfilter_x[None, ...] + \
                  np.reshape(self.sphere.vertices[:, 1], (-1, 1, 1, 1)) * self.wfilter_y[None, ...] + \
                  np.reshape(self.sphere.vertices[:, 2], (-1, 1, 1, 1)) * self.wfilter_z[None, ...]
        g4 = (16.0*x_prime**4 - 48*x_prime**2 + 12) * self._exp()[None, ...]
        return g4

    def _H4(self):
        x_prime = np.reshape(self.sphere.vertices[:, 0], (-1, 1, 1, 1)) * self.wfilter_x[None, ...] + \
                  np.reshape(self.sphere.vertices[:, 1], (-1, 1, 1, 1)) * self.wfilter_y[None, ...] + \
                  np.reshape(self.sphere.vertices[:, 2], (-1, 1, 1, 1)) * self.wfilter_z[None, ...]
        h4 = (5.10495*x_prime**5 - 38.29678*x_prime**3 + 36.70429*x_prime) * self._exp()[None, ...]
        return h4

    def _exp(self):
        return np.exp(-(self.wfilter_x**2 + self.wfilter_y**2 + self.wfilter_z**2))

    def _G2a(self):
        basis = self.normalization_G2 * (2.0*self.wfilter_x**2 - 1.0) * self._exp()
        kappa = self.sphere.vertices[:, 0]**2  # alpha^2
        return kappa, basis
    
    def _G2b(self):
        basis = self.normalization_G2 * (2.0*self.wfilter_x*self.wfilter_y) * self._exp()
        kappa = 2.0*self.sphere.vertices[:, 0]*self.sphere.vertices[:, 1]  # 2*alpha*beta
        return kappa, basis
    
    def _G2c(self):
        basis = self.normalization_G2 * (2.0 * self.wfilter_y**2 - 1) * self._exp()
        kappa = self.sphere.vertices[:, 1]**2  # beta**2
        return kappa, basis
    
    def _G2d(self):
        basis = self.normalization_G2 * (2.0*self.wfilter_x*self.wfilter_z) * self._exp()
        kappa = 2.0 * self.sphere.vertices[:, 0] * self.sphere.vertices[:, 2]  # 2alpha*gamma
        return kappa, basis

    def _G2e(self):
        basis = self.normalization_G2 * (2.0*self.wfilter_y*self.wfilter_z) * self._exp()
        kappa = 2.0 * self.sphere.vertices[:, 1] * self.sphere.vertices[:, 2]  # 2beta*gamma
        return kappa, basis

    def _G2f(self):
        basis = self.normalization_G2 * (2.0*self.wfilter_z**2 - 1) * self._exp()
        kappa = self.sphere.vertices[:, 2]**2  # gamma**2
        return kappa, basis

    def _H2a(self):
        basis = self.normalization_H2 * (self.wfilter_x**3 - 2.254*self.wfilter_x) * self._exp()
        kappa = self.sphere.vertices[:, 0]**3
        return kappa, basis

    def _H2b(self):
        basis = self.normalization_H2 * self.wfilter_y * (self.wfilter_x**2 - 0.751333) * self._exp()
        kappa = 3.0 * self.sphere.vertices[:, 0]**2 * self.sphere.vertices[:, 1]
        return kappa, basis

    def _H2c(self):
        basis = self.normalization_H2 * self.wfilter_x * (self.wfilter_y**2 - 0.751333) * self._exp()
        kappa = 3.0 * self.sphere.vertices[:, 0] * self.sphere.vertices[:, 1]**2
        return kappa, basis

    def _H2d(self):
        basis = self.normalization_H2 * self.wfilter_y * (self.wfilter_y**2 - 2.254) * self._exp()
        kappa = self.sphere.vertices[:, 1]**3
        return kappa, basis

    def _H2e(self):
        basis = self.normalization_H2 * self.wfilter_z * (self.wfilter_x**2 - 0.751333) * self._exp()
        kappa = 3.0 * self.sphere.vertices[:, 0]**2 * self.sphere.vertices[:, 2]
        return kappa, basis

    def _H2f(self):
        basis = self.normalization_H2 * self.wfilter_x * self.wfilter_y * self.wfilter_z * self._exp()
        kappa = 6.0 * self.sphere.vertices[:, 0] * self.sphere.vertices[:, 1] * self.sphere.vertices[:, 2]
        return kappa, basis

    def _H2g(self):
        basis = self.normalization_H2 * self.wfilter_z * (self.wfilter_y**2 - 0.751333) * self._exp()
        kappa = 3.0 * self.sphere.vertices[:, 1]**2 * self.sphere.vertices[:, 2]
        return kappa, basis

    def _H2h(self):
        basis = self.normalization_H2 * self.wfilter_x * (self.wfilter_z**2 - 0.751333) * self._exp()
        kappa =3.0 * self.sphere.vertices[:, 0] * self.sphere.vertices[:, 2]**2
        return kappa, basis

    def _H2i(self):
        basis = self.normalization_H2 * self.wfilter_y * (self.wfilter_z**2 - 0.751333) * self._exp()
        kappa = 3.0 * self.sphere.vertices[:, 1] * self.sphere.vertices[:, 2]**2
        return kappa, basis

    def _H2j(self):
        basis = self.normalization_H2 * self.wfilter_z * (self.wfilter_z**2 - 2.254) * self._exp()
        kappa = self.sphere.vertices[:, 2]**3
        return kappa, basis

    def _derivative_of_gaussian_quadrature_pair(self, values):
        """
        Compute the quadrature of Gaussian derivatives of specified order.

        Parameters
        ----------
        values : ndarray
            Neighborhood values to process.
        derivative_order : int, optional
            Order of the derivative (2 or 4). Default is 2.

        Returns
        -------
        ndarray
            Quadrature response combining G and H filter responses.
        """
        if self.derivative_order == 2:
            g_response = np.sum(self.G2_filter.dot(self.G2_kappas) * values[..., None], axis=(0, 1, 2))
            h_response = np.sum(self.H2_filter.dot(self.H2_kappas) * values[..., None], axis=(0, 1, 2))
        elif self.derivative_order == 4:
            g_response = np.sum(self.G4_filters * values[None, ...], axis=(1, 2, 3))
            h_response = np.sum(self.H4_filters * values[None, ...], axis=(1, 2, 3))
        else:
            raise ValueError(f"derivative_order must be 2 or 4, got {self.derivative_order}")
        
        return g_response ** 2 + h_response ** 2

    def _get_sf(self, pos):
        pos_neighbours = self.neighbour_offsets + np.asarray(pos).reshape((3,)) 
        neighbourhood = np.zeros(self.neighbour_offsets.shape[:3])
        for ii in range(neighbourhood.shape[0]):
            for jj in range(neighbourhood.shape[1]):
                for kk in range(neighbourhood.shape[2]):
                    neighbourhood[ii, jj, kk] =\
                        self.datavolume.get_value_at_coordinate(
                            *pos_neighbours[ii, jj, kk], space=self.space, origin=self.origin
                            )

        # normalize the patch
        neighbourhood -= neighbourhood.min()
        if neighbourhood.max() > 0:
            neighbourhood /= neighbourhood.max()

        # Compute SF at pos using sphere
        sf = self._derivative_of_gaussian_quadrature_pair(neighbourhood)

        # SF to SH, followed by FRT, back to SF
        odf = sf.T.dot(self.B_inv).dot(self.FRT).dot(self.B).reshape((-1, 1))
        odf_max = np.max(odf)
        if odf_max > 0:
            odf /= odf_max
        odf = odf ** 2  # squared to sharpen principal directions

        self.iteration += 1
        return odf

    def prepare_forward(self, seeding_pos, random_generator):
        """
        Prepare information necessary at the first point of the
        streamline for forward propagation: v_in and any other information
        necessary for the self.propagate method.

        About **v_in**, it is used for two things:

        - To sample the next direction based on _sample_next_direction method.
            Ex, with fODF, it defines a cone theta of accepable directions.
        - If no valid next dir are found, continue straight.

        Parameters
        ----------
        seeding_pos: tuple(x,y,z)
            The seeding position. Important, position must be in the same space
            and origin as self.space, self.origin!
        random_generator: numpy Generator

        Returns
        -------
        v_in: TrackingDirection
            The "fake" previous direction at first step. Could be None if your
            propagator can propagate without knowledge of previous direction.
            Return PropagationStatus.Error if no good tracking direction can be
            set at current seeding position.
        """
        # Sampling on the SF values (no matter if general algo is det or prob)
        # with a different threshold than usual (sf_threshold_init).
        # So the initial step's propagation will be in a cone theta around a
        # "more probable" peak.
        self.iteration = 0
        sf = self._get_sf(seeding_pos)
        sf[sf < self.sf_threshold_init] = 0
        self.line_rng_generator = random_generator

        if np.sum(sf) > 0:
            if self.algo == 'det':
                ind = np.argmax(sf)
            else:
                ind = sample_distribution(sf, self.line_rng_generator)
            return TrackingDirection(self.dirs[ind], ind)

        # Else: sf at current position is smaller than acceptable threshold in
        # all directions.
        return PropagationStatus.ERROR
    

class MicroscopyPeakPropagator(AbstractPropagator):
    """
    Propagator on microscopy peaks extracted from directional data.
    Uses peaks detected from microscopy imaging to guide streamline tracking.
    """
    def __init__(self, datavolume, step_size, rk_order, window_halfwidth, theta, space, origin):
        """
        Parameters
        ----------
        datavolume: scilpy.image.volume_space_management.DataVolume
            Trackable Dataset object containing peak directions.
        step_size: float
            The step size for tracking. Important: step size should be in the
            same units as the space of the tracking!
        rk_order: int
            Order for the Runge Kutta integration.
        space: dipy Space
            Space of the streamlines during tracking.
        origin: dipy Origin
            Origin of the streamlines during tracking. All coordinates received
            in the propagator's methods will be expected to respect
            that origin.
        window_halfwidth: int
            Half-width of the window for peak detection in microscopy data.
        """
        super().__init__(datavolume, step_size, rk_order, space, origin)

        if self.space != Space.VOX:
            raise NotImplementedError(
                "This version of the propagator is not ready to work "
                "in VOXMM/RASMM space.")

        self.theta = theta
        self.window_halfwidth = np.sort(window_halfwidth)
        self.max_halfwidth = self.window_halfwidth[-1]
        self.samples_grid = np.stack(
            np.meshgrid(*[np.arange(-self.max_halfwidth, self.max_halfwidth+1)
                          for _ in range(3)], indexing='ij'), axis=-1)
        self.wfilters = []
        for w in window_halfwidth:
           self.wfilters.append(self._scaled_derivatives(w))
        self.wfilters = np.asarray(self.wfilters)
        import nibabel as nib
        for i in range(self.wfilters.shape[0]):
            nib.save(nib.Nifti1Image(np.moveaxis(self.wfilters[i], 0, -1).astype(np.float32), np.eye(4)),
                     f'derivatives_{i}.nii.gz')
        print(self.wfilters.shape)

    def _scaled_derivatives(self, width):
        # convert width to sigma
        sigma = width / 3.0
        x, y ,z = np.meshgrid(*[np.arange(-self.max_halfwidth, self.max_halfwidth+1) for _ in range(3)],
                              indexing='ij')
        gaussian = np.exp(-(x**2+y**2+z**2)/2.0/sigma**2)
        # dx = dx_multiplier * gaussian
        dx_multiplier = -x / sigma**2
        dy_multiplier = -y / sigma**2
        dz_multiplier = -z / sigma**2
        # all second order derivatives
        dxdx = (dx_multiplier * dx_multiplier - 1.0/sigma**2) * gaussian
        dxdy = dx_multiplier * dy_multiplier * gaussian
        dxdz = dx_multiplier * dz_multiplier * gaussian
        dydy = (dy_multiplier * dy_multiplier - 1.0/sigma**2) * gaussian
        dydz = dy_multiplier * dz_multiplier * gaussian
        dzdz = (dz_multiplier * dz_multiplier - 1.0/sigma**2) * gaussian
        all_derivatives = np.stack([dxdx, dxdy, dxdz, dydy, dydz, dzdz], axis=0)
        all_derivatives /= np.sum(all_derivatives**2, axis=(1, 2, 3), keepdims=True)
        return sigma * all_derivatives

    def prepare_forward(self, seeding_pos, random_generator):
        """
        Prepare information necessary at the first point of the
        streamline for forward propagation.

        Parameters
        ----------
        seeding_pos: tuple(x,y,z)
            The seeding position.
        random_generator: numpy Generator

        Returns
        -------
        tracking_info: Any
            Any tracking information necessary for the propagation.
        """
        self.line_rng_generator = random_generator
        directions, score = self._get_possible_directions(seeding_pos)
        if score is not None:
            return TrackingDirection(directions[np.argmax(score)])
        return PropagationStatus.ERROR

    def _sample_next_direction(self, pos, v_in):
        """
        Chooses a next tracking direction from all possible directions offered
        by the tracking field.

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        -------
        direction: ndarray (3,)
            A valid tracking direction. None if no valid direction is found.
        """
        directions, score = self._get_possible_directions(pos)
        if score is not None:
            cosangles = directions.dot(np.reshape(v_in, (3, 1)))
            if np.any(np.abs(cosangles) > np.cos(self.theta)):
                argmax = np.argmax(np.abs(cosangles))
                best_cos = cosangles[argmax]
                best_dir = directions[argmax]
                # best_dir might be flipped wrt v_in, flip direction is cosangles < 0
                return TrackingDirection(best_dir) if best_cos > 0 else TrackingDirection(-best_dir)

        # if no valid direction is within the tracking cone return None
        return None

    def _get_possible_directions(self, pos):
        pos_neighbours = self.samples_grid + np.asarray(pos).reshape((3,))
        neighbourhood = np.zeros(self.samples_grid.shape[:3])
        for ii in range(neighbourhood.shape[0]):
            for jj in range(neighbourhood.shape[1]):
                for kk in range(neighbourhood.shape[2]):
                    neighbourhood[ii, jj, kk] =\
                        self.datavolume.get_value_at_coordinate(
                            *pos_neighbours[ii, jj, kk], space=self.space, origin=self.origin
                            )

        # neighbourhood shape is (nx, ny, nz) and wfilters is (nscales, nx, ny, nz, 3)
        # this is the x-y-z derivatives at each scale (nscales, 3)
        derivatives = np.sum(neighbourhood[None, None, ::-1, ::-1, ::-1] * self.wfilters, axis=(2, 3, 4))

        # Compute matrix of shape (nscales, 3, 3) where each element [k, i, j] is derivatives[k, i].dot(derivatives[k, j])
        hessian = np.array([[derivatives[:, 0], derivatives[:, 1], derivatives[:, 2]],
                            [derivatives[:, 1], derivatives[:, 3], derivatives[:, 4]],
                            [derivatives[:, 2], derivatives[:, 4], derivatives[:, 5]]])
        hessian = np.moveaxis(hessian, -1, 0)

        # evals has shape (nscales, 3) and evecs has shape (nscales, 3, 3)
        evals, evecs = np.linalg.eigh(hessian)
        # sort evals/evecs by absolute value
        evals_argsort = np.argsort(np.abs(evals), axis=-1)
        evals = np.take_along_axis(evals, evals_argsort, axis=-1)
        principal_dirs = np.take_along_axis(evecs, evals_argsort[:, None, :], axis=2)[..., 0]
        is_valid_dir = (evals[:, 1] < 0) & (evals[:, 2] < 0)
        if np.any(is_valid_dir):
            # keep only directions corresponding to bright structure over dark background
            principal_dirs = principal_dirs[is_valid_dir]
            evals = evals[is_valid_dir]
            score = np.exp(-1/2 * evals[:, 0]**2 / np.abs(evals[:, 0]).max()**2)
            return principal_dirs, score

        return [], None


class FibertubePropagator(AbstractPropagator):
    """
    Simplified propagator for using fibertube data. It is probabilistic and
    uses the volume of intersection between fibertube segments and the
    blurring sphere as a random distribution for picking a segment. This
    segment is then used as the propagation direction.

    This propagator expects an array of possible directions and their random
    distribution. If using an ftODF (the same directions, but expressed as a
    spherical function), the ODFPropagator should be used.
    """
    def __init__(self, datavolume: FibertubeDataVolume, step_size, rk_order,
                 theta, space, origin):
        """"
        Parameters
        ----------
        datavolume: FibertubeDataVolume
            Trackable fibertube dataset object.
        step_size: float
            The step size for tracking. Important: step size should be in the
            same units as the space of the tracking!
        rk_order: int
            Order for the Runge Kutta integration.
        theta: float
            Maximum angle (radians) between two steps.
        space: dipy Space
            Space of the streamlines during tracking. value.
        origin: dipy Origin
            Origin of the streamlines during tracking. All coordinates
            received in the propagator's methods will be expected to respect
            that origin.

        A note on space and origin: All coordinates received in the
        propagator's methods will be expected to respect those values. Tracker
        will verify that the propagator has the same internal values as itself.
        """

        if not (rk_order == 1 or rk_order == 2 or rk_order == 4):
            raise ValueError("Invalid runge-kutta order. Is " +
                             str(rk_order) + ". Choices : 1, 2, 4")

        self.datavolume = datavolume
        self.step_size = step_size
        self.rk_order = rk_order
        self.theta = theta
        self.space = space
        self.origin = origin
        self.normalize_directions = True
        # Will be reset at each new streamline.
        self.line_rng_generator = None

    def reset_data(self, new_data=None):
        return super().reset_data(new_data)

    def prepare_forward(self, seeding_pos, random_generator):
        direction = self.datavolume.get_absolute_direction(*seeding_pos)

        # Validate seeding within a fibertube.
        if direction is None:
            return PropagationStatus.ERROR

        self.line_rng_generator = random_generator

        return TrackingDirection(direction)

    def prepare_backward(self, line, forward_dir):
        return super().prepare_backward(line, forward_dir)

    def finalize_streamline(self, last_pos, v_in):
        return super().finalize_streamline(last_pos, v_in)

    def propagate(self, line, v_in):
        return super().propagate(line, v_in)

    def _sample_next_direction(self, pos, v_in):
        directions, volumes = self._get_possible_next_dirs(pos, v_in)

        # Sampling one.
        if np.sum(volumes) > 0:
            v_out = directions[
                sample_distribution(volumes, self.line_rng_generator)]
            return v_out
        return None

    def _get_possible_next_dirs(self, pos, v_in):
        directions, volumes = (
            self.datavolume.get_value_at_coordinate(*pos, self.space,
                                                    self.origin))

        # Angle threshold
        valid_dirs = []
        valid_volumes = []

        for i, dir in enumerate(directions):
            num = np.dot(v_in, dir)
            cosine = num / (np.linalg.norm(v_in) *
                            np.linalg.norm(dir))

            # Flip direction if facing the wrong way
            if cosine < 0:
                cosine = abs(cosine)
                dir = -dir

            cosine = np.clip(cosine, -1, 1)

            if (np.arccos(cosine) > self.theta):
                continue

            valid_dirs.append(dir)
            valid_volumes.append(volumes[i])

        valid_dirs = np.array(valid_dirs)
        valid_volumes = np.array(valid_volumes)

        return valid_dirs, valid_volumes

