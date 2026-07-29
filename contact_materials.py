"""Umgebungsspezifische PhysX-Materialien fuer Parameterstudien."""

from project_config import CONTACT, EnvironmentParameters


def _validate_friction(static, dynamic, label):
    if static < 0.0 or dynamic < 0.0:
        raise ValueError(f"{label}: Reibung darf nicht negativ sein.")
    if dynamic > static:
        raise ValueError(
            f"{label}: dynamische Reibung darf statische nicht uebersteigen."
        )


def validate_parameters(parameters: EnvironmentParameters):
    _validate_friction(
        parameters.peg_socket_static_friction,
        parameters.peg_socket_dynamic_friction,
        "Peg/Buchse",
    )
    _validate_friction(
        parameters.gripper_peg_static_friction,
        parameters.gripper_peg_dynamic_friction,
        "Greifer/Peg",
    )


def _define(
    stage,
    path,
    static_friction,
    dynamic_friction,
    stiffness=None,
    damping=None,
):
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction))
    api.CreateRestitutionAttr(float(CONTACT.restitution))
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr().Set(PhysxSchema.Tokens.max)
    if stiffness is not None:
        physx.CreateCompliantContactStiffnessAttr().Set(float(stiffness))
        physx.CreateCompliantContactDampingAttr().Set(float(damping))
    return material


def create_environment_materials(stage, root_path, parameters):
    validate_parameters(parameters)
    material_root = f"{root_path}/PhysicsMaterials"
    return {
        "peg": _define(
            stage,
            f"{material_root}/Peg",
            parameters.peg_socket_static_friction,
            parameters.peg_socket_dynamic_friction,
        ),
        "socket": _define(
            stage,
            f"{material_root}/Socket",
            parameters.peg_socket_static_friction,
            parameters.peg_socket_dynamic_friction,
            CONTACT.peg_socket_contact_stiffness_n_per_m,
            CONTACT.peg_socket_contact_damping_ns_per_m,
        ),
        "gripper": _define(
            stage,
            f"{material_root}/Gripper",
            parameters.gripper_peg_static_friction,
            parameters.gripper_peg_dynamic_friction,
            CONTACT.gripper_contact_stiffness_n_per_m,
            CONTACT.gripper_contact_damping_ns_per_m,
        ),
    }


def bind_physics_material(stage, prim_path, material):
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Materialziel fehlt: {prim_path}")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

