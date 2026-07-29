"""Vollstaendige 6D-D6-Nachgiebigkeit mit MuJoCo-Achsenmapping."""

import numpy as np

from project_config import ISAAC_COMPLIANCE, EnvironmentParameters


def angular_drive_coefficient_per_degree(value_per_radian):
    """USD angular DriveAPI uses degrees, MuJoCo uses radians."""

    return float(value_per_radian) * np.pi / 180.0


def compute_compliance_state(
    reference_position,
    body_position,
    reference_velocity,
    body_velocity,
    parameters: EnvironmentParameters,
):
    reference_position = np.asarray(reference_position, dtype=float)
    body_position = np.asarray(body_position, dtype=float)
    reference_velocity = np.asarray(reference_velocity, dtype=float)
    body_velocity = np.asarray(body_velocity, dtype=float)
    deflection = body_position - reference_position
    relative_velocity = body_velocity - reference_velocity
    stiffness = np.array(
        [
            parameters.stiffness_x_n_per_m,
            parameters.stiffness_y_n_per_m,
            parameters.stiffness_z_n_per_m,
        ],
        dtype=float,
    )
    damping = np.array(
        [
            parameters.damping_x_ns_per_m,
            parameters.damping_y_ns_per_m,
            parameters.damping_z_ns_per_m,
        ],
        dtype=float,
    )
    spring_force = -stiffness * deflection
    damping_force = -damping * relative_velocity
    total_force = np.clip(
        spring_force + damping_force,
        -ISAAC_COMPLIANCE.max_drive_force_n,
        ISAAC_COMPLIANCE.max_drive_force_n,
    )
    return {
        "deflection_m": deflection,
        "relative_velocity_m_per_s": relative_velocity,
        "spring_force_n": spring_force,
        "damping_force_n": damping_force,
        "total_force_n": total_force,
    }


class StructuredComplianceJoint:
    """D6-Joint: drei Translationen und drei endliche Rotationsdrives.

    Anders als im vorherigen Isaac-Schritt sind die Rotationen nicht gesperrt:
    Die MuJoCo-Basis besitzt 100/100/1 N*m/rad und 1 N*m*s/rad.
    """

    def __init__(
        self,
        stage,
        root_path,
        reference_body_path,
        compliant_body_path,
        parameters: EnvironmentParameters,
    ):
        from pxr import Gf, Sdf, UsdPhysics

        self._UsdPhysics = UsdPhysics
        self.parameters = parameters
        self.joint_path = f"{root_path}/StructuredComplianceJoint"
        self.joint = UsdPhysics.Joint.Define(stage, self.joint_path)
        self.joint.CreateBody0Rel().SetTargets([Sdf.Path(reference_body_path)])
        self.joint.CreateBody1Rel().SetTargets([Sdf.Path(compliant_body_path)])
        zero = Gf.Vec3f(0.0, 0.0, 0.0)
        identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        self.joint.CreateLocalPos0Attr().Set(zero)
        self.joint.CreateLocalPos1Attr().Set(zero)
        self.joint.CreateLocalRot0Attr().Set(identity)
        self.joint.CreateLocalRot1Attr().Set(identity)
        prim = self.joint.GetPrim()

        # MuJoCo besitzt keine expliziten Translationsgrenzen. Diese weiten
        # Limits sind nur ein numerischer Not-Aus fuer eine verlorene Montage.
        for axis in (
            UsdPhysics.Tokens.transX,
            UsdPhysics.Tokens.transY,
            UsdPhysics.Tokens.transZ,
        ):
            limit = UsdPhysics.LimitAPI.Apply(prim, axis)
            limit.CreateLowAttr(-ISAAC_COMPLIANCE.max_linear_deflection_m)
            limit.CreateHighAttr(ISAAC_COMPLIANCE.max_linear_deflection_m)

        translation = (
            (
                UsdPhysics.Tokens.transX,
                parameters.stiffness_x_n_per_m,
                parameters.damping_x_ns_per_m,
            ),
            (
                UsdPhysics.Tokens.transY,
                parameters.stiffness_y_n_per_m,
                parameters.damping_y_ns_per_m,
            ),
            (
                UsdPhysics.Tokens.transZ,
                parameters.stiffness_z_n_per_m,
                parameters.damping_z_ns_per_m,
            ),
        )
        rotation = (
            (
                UsdPhysics.Tokens.rotX,
                parameters.stiffness_rot_x_nm_per_rad,
                parameters.damping_rot_x_nms_per_rad,
            ),
            (
                UsdPhysics.Tokens.rotY,
                parameters.stiffness_rot_y_nm_per_rad,
                parameters.damping_rot_y_nms_per_rad,
            ),
            (
                UsdPhysics.Tokens.rotZ,
                parameters.stiffness_rot_z_nm_per_rad,
                parameters.damping_rot_z_nms_per_rad,
            ),
        )
        self.drives = {}
        for axis, stiffness, damping in translation:
            self._create_drive(
                prim,
                axis,
                stiffness,
                damping,
                ISAAC_COMPLIANCE.max_drive_force_n,
            )
        for axis, stiffness, damping in rotation:
            self._create_drive(
                prim,
                axis,
                angular_drive_coefficient_per_degree(stiffness),
                angular_drive_coefficient_per_degree(damping),
                ISAAC_COMPLIANCE.max_drive_torque_nm,
            )

    def _create_drive(self, prim, axis, stiffness, damping, max_force):
        drive = self._UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr(self._UsdPhysics.Tokens.force)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateStiffnessAttr(float(stiffness))
        drive.CreateDampingAttr(float(damping))
        drive.CreateMaxForceAttr(float(max_force))
        self.drives[str(axis)] = drive

    def measure(
        self,
        reference_position,
        body_position,
        reference_velocity,
        body_velocity,
    ):
        return compute_compliance_state(
            reference_position,
            body_position,
            reference_velocity,
            body_velocity,
            self.parameters,
        )
