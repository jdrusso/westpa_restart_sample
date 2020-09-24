import os
import errno
import time
import numpy as np
from west.propagators import WESTPropagator
from west import Segment
from west.states import BasisState, InitialState

import simtk.openmm.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as units


import sys
import logging

logging.getLogger("h5py").setLevel(logging.ERROR)

log = logging.getLogger(__name__)
log.debug("loading module %r" % __name__)

pcoord_len = 5
pcoord_dtype = np.float32


class Timer:
    def __init__(self, label):
        self.label = label
        self.start_time = time.time()

    def check(self, label=None):
        if label is None:
            label = self.label

        print("%s | Time elapsed: %.1f" % (label, time.time() - self.start_time))
        sys.stdout.flush()


class OpenMMPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(OpenMMPropagator, self).__init__(rc)

        self.runfile = "pos.gro"

        init_timer = Timer("init")
        self.pcoord_len = pcoord_len
        self.pcoord_dtype = pcoord_dtype
        self.pcoord_ndim = 1

        self.BNZ_indices = range(2603, 2615)

        self.basis_coordinates = np.array(
            [[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]], dtype=pcoord_dtype
        )

        config = self.rc.config

        # Validate configuration
        for key in [
            ("west", "openmm", "system", "file"),
            ("west", "openmm", "integrator", "file"),
            ("west", "openmm", "integrator", "steps_per_tau"),
            ("west", "openmm", "integrator", "steps_per_write"),
            ("west", "openmm", "integrator", "steps_per_coord_write"),
            ("west", "openmm", "platform", "name"),
            ("west", "data", "data_refs", "initial_state"),
        ]:
            config.require(key)

        self.initial_state_ref_template = config[
            "west", "data", "data_refs", "initial_state"
        ]

        system_xml_file = config["west", "openmm", "system", "file"]
        self.integrator_xml_file = config["west", "openmm", "integrator", "file"]

        self.steps_per_tau = config["west", "openmm", "integrator", "steps_per_tau"]
        self.steps_per_write = config["west", "openmm", "integrator", "steps_per_write"]
        self.steps_per_coord_write = int(
            config["west", "openmm", "integrator", "steps_per_coord_write"]
        )
        self.nblocks = (self.steps_per_tau // self.steps_per_write) + 1

        platform_name = config["west", "openmm", "platform", "name"] or "Reference"
        # config_platform_properties = config['west', 'openmm', 'platform', 'properties'] or {}

        # Set up OpenMM
        with open(system_xml_file, "r") as f:
            # NOTE: calling the system self.system causes a namespace collision in the propagator
            self.mmsystem = openmm.XmlSerializer.deserialize(f.read())

        self.basis_coordinates = app.PDBFile(self.runfile).getPositions(asNumpy=True)
        self.initial_positions = self.basis_coordinates

        self.platform = openmm.Platform.getPlatformByName(platform_name)
        self.platform_properties = {}

        self.temperature = 300 * units.kelvin

        # print("Initialization took %.4f seconds" % (time.time() - init_time))
        init_timer.check()

    # Result should be returned in nanometers, there's a unit conversion
    #   elsewhere that depends on that.
    @staticmethod
    def dist(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    @staticmethod
    def makepath(
        template,
        template_args=None,
        expanduser=True,
        expandvars=True,
        abspath=False,
        realpath=False,
    ):
        template_args = template_args or {}
        path = template.format(**template_args)
        if expandvars:
            path = os.path.expandvars(path)
        if expanduser:
            path = os.path.expanduser(path)
        if realpath:
            path = os.path.realpath(path)
        if abspath:
            path = os.path.abspath(path)
        path = os.path.normpath(path)
        return path

    @staticmethod
    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    # Only used for initial states!
    def get_pcoord(self, state):

        if isinstance(state, BasisState):
            coords = self.basis_coordinates.copy()
        elif isinstance(state, InitialState):
            template_args = {"initial_state": state}
            istate_data_ref = self.makepath(
                self.initial_state_ref_template, template_args
            )

            coords = np.loadtxt(istate_data_ref)
        else:
            raise TypeError("state must be BasisState or InitialState")

        state.pcoord = self.dist(coords[0, :], coords[1, :])

    # @profile
    def propagate(self, segments):

        prop_timer = Timer("propagate")
        starttime = time.time()

        platform = openmm.Platform.getPlatformByName("OpenCL")
        prop = {}

        with open(self.integrator_xml_file, "r") as f:
            integrator = openmm.XmlSerializer.deserialize(f.read())

        context = openmm.Context(self.mmsystem, integrator, platform, prop)
        # prop_timer.check("finished context creation")

        for segment in segments:
            # print("Starting segment")
            sys.stdout.flush()
            sys.stderr.flush()
            seg_timer = Timer("segment")

            # Set up arrays to hold trajectory data for pcoords, coordinates and velocities
            pcoords = np.empty((self.nblocks, 1))
            pcoords[0] = segment.pcoord[0]

            coordinates = np.empty((self.nblocks, self.mmsystem.getNumParticles(), 3))
            velocities = np.empty((self.nblocks, self.mmsystem.getNumParticles(), 3))

            # Get initial coordinates and velocities from restarts or initial state
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                # Get restart data
                assert "restart_coord" in segment.data
                assert "restart_veloc" in segment.data

                coordinates[0] = segment.data["restart_coord"]
                velocities[0] = segment.data["restart_veloc"]

                initial_coords = units.Quantity(
                    segment.data["restart_coord"], units.nanometer
                )
                initial_velocs = units.Quantity(
                    segment.data["restart_veloc"], units.nanometer / units.picosecond
                )

                context.setPositions(initial_coords)
                context.setVelocities(initial_velocs)

                del segment.data["restart_coord"]
                del segment.data["restart_veloc"]

            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                initial_state = self.initial_states[segment.initial_state_id]

                assert initial_state.istate_type == InitialState.ISTATE_TYPE_GENERATED

                # Load coordinates coresponding to the initial state
                # new_template_args = {"initial_state": initial_state}
                # istate_data_ref = self.makepath(
                #     self.initial_state_ref_template, new_template_args
                # )

                # Set up context for this segment
                ips = app.PDBFile(self.runfile).getPositions(asNumpy=True)
                context.setPositions(ips)
                context.setVelocitiesToTemperature(self.temperature)

                state = context.getState(
                    getPositions=True, getVelocities=True, enforcePeriodicBox=True
                )
                coordinates[0] = state.getPositions(asNumpy=True)
                velocities[0] = state.getVelocities(asNumpy=True)

            # seg_timer.check("Loaded data for segments")

            # Run dynamics
            for istep in range(1, self.nblocks):
                integrator.step(self.steps_per_write)

                state = context.getState(
                    getPositions=True, getVelocities=True, enforcePeriodicBox=True
                )

                coordinates[istep] = state.getPositions(asNumpy=True)
                velocities[istep] = state.getVelocities(asNumpy=True)

                # Compute progress coordinate
                # pcoords[istep] = 10.0 * \
                # self.dist(coordinates[istep,0,:], coordinates[istep,1,:])

                cur_pos = np.array([coordinates[istep, x, :] for x in self.BNZ_indices])

                ref_pos = np.array(
                    [self.basis_coordinates[x, :]._value for x in self.BNZ_indices]
                )

                rmsd = np.sqrt(np.mean(np.square(cur_pos - ref_pos)))
                # print("RMSD is ", end='')
                # print(rmsd)

                pcoords[istep] = 10.0 * rmsd

            # Finalize segment trajectory
            segment.pcoord = pcoords[...].astype(pcoord_dtype)
            print("Pcoord: ", end="")
            print(segment.pcoord[0])

            segment.data["coord"] = coordinates[:: self.steps_per_coord_write]
            segment.data["veloc"] = velocities[:: self.steps_per_coord_write]

            segment.status = Segment.SEG_STATUS_COMPLETE

            segment.walltime = time.time() - starttime
            segment.cputime = segment.walltime
            seg_timer.check(
                "Finished propagation. contd?: %r"
                % (segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES)
            )
            # print("CPUTime for this segment was %f" % segment.cputime)
            # sys.stdout.flush()

        return segments

    def gen_istate(self, basis_state, initial_state):
        """Generate a new initial state from the given basis state."""

        initial_state.pcoord = np.array([0], dtype=pcoord_dtype)
        initial_state.istate_status = initial_state.ISTATE_STATUS_PREPARED

        return initial_state
