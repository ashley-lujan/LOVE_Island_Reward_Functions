#!/usr/bin/env python3
"""Generate a minimal cartpole.usda (ASCII USD) using plain file I/O — no pxr needed.

Run on the HOST (no container required):
  python scripts/create_cartpole_usd.py --output /path/to/cartpole.usda

Robot structure (matches the IsaacGym / Isaac Lab cartpole joint naming):
  /Robot  (ArticulationRoot, isFixedBase=true)
    /slider              — articulation root link, world-fixed via isFixedBase
    /cart                — slides along X (prismatic joint: slider_to_cart)
    /pole                — rotates around Y (revolute joint: cart_to_pole)
    /slider_to_cart      — PhysicsPrismaticJoint (concrete type), force-driven
    /cart_to_pole        — PhysicsRevoluteJoint (concrete type), passive

Critical: Use concrete USD types (PhysicsPrismaticJoint, PhysicsRevoluteJoint),
NOT "PhysicsJoint + API schema". The concrete types create single-DOF joints in
PhysX articulations whose names match the prim name (e.g. "slider_to_cart").
The API-schema approach creates D6 joints whose DOF names get :0/:1/:2 suffixes.
"""

import argparse
import os

USDA_TEMPLATE = """\
#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
    defaultPrim = "Robot"
)

def PhysicsScene "physicsScene"
{
    vector3f physics:gravityDirection = (0, 0, -1)
    float physics:gravityMagnitude = 9.81
}

# isFixedBase=true anchors the slider to the world.
# PhysX does not support kinematic bodies inside articulations.
def Xform "Robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxSchema:PhysxArticulationAPI"]
)
{
    bool physxArticulation:isFixedBase = true
    double3 xformOp:translate = (0, 0, 2)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    # -------------------------------------------------------------------
    # Slider — root link, world-fixed via isFixedBase.
    # Not kinematic: kinematic bodies break PhysX articulations.
    # -------------------------------------------------------------------
    def Xform "slider" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.01
    }

    # -------------------------------------------------------------------
    # Cart — slides along X axis
    # -------------------------------------------------------------------
    def Cube "cart" (
        prepend apiSchemas = [
            "PhysicsRigidBodyAPI",
            "PhysicsCollisionAPI",
            "PhysicsMassAPI",
            "PhysxSchema:PhysxRigidBodyAPI"
        ]
    )
    {
        double size = 1
        float physics:mass = 1.0
        float3 xformOp:scale = (0.4, 0.2, 0.1)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }

    # -------------------------------------------------------------------
    # Pole — rotates around Y axis
    # -------------------------------------------------------------------
    def Capsule "pole" (
        prepend apiSchemas = [
            "PhysicsRigidBodyAPI",
            "PhysicsCollisionAPI",
            "PhysicsMassAPI",
            "PhysxSchema:PhysxRigidBodyAPI"
        ]
    )
    {
        uniform token axis = "Z"
        float height = 0.8
        float radius = 0.02
        float physics:mass = 0.1
        double3 xformOp:translate = (0, 0, 0.5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    # -------------------------------------------------------------------
    # Prismatic joint: slider -> cart  (DOF name "slider_to_cart")
    # Use concrete type PhysicsPrismaticJoint — creates a single-DOF joint
    # in the PhysX articulation with DOF name == prim name (no :0 suffix).
    # -------------------------------------------------------------------
    def PhysicsPrismaticJoint "slider_to_cart" (
        prepend apiSchemas = [
            "PhysicsDriveAPI:linear",
            "PhysxSchema:PhysxJointAPI"
        ]
    )
    {
        rel physics:body0 = </Robot/slider>
        rel physics:body1 = </Robot/cart>
        token physics:axis = "X"
        float physics:lowerLimit = -3.0
        float physics:upperLimit = 3.0

        token drive:linear:physics:type = "force"
        float drive:linear:physics:maxForce = 400.0
        float drive:linear:physics:stiffness = 0.0
        float drive:linear:physics:damping = 0.0
    }

    # -------------------------------------------------------------------
    # Revolute joint: cart -> pole  (DOF name "cart_to_pole")
    # Use concrete type PhysicsRevoluteJoint — single-DOF around Y.
    # -------------------------------------------------------------------
    def PhysicsRevoluteJoint "cart_to_pole" (
        prepend apiSchemas = [
            "PhysxSchema:PhysxJointAPI"
        ]
    )
    {
        rel physics:body0 = </Robot/cart>
        rel physics:body1 = </Robot/pole>
        token physics:axis = "Y"

        float3 physics:localPos0 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        float3 physics:localPos1 = (0, 0, -0.5)
        quatf physics:localRot1 = (1, 0, 0, 0)
    }
}
"""


def create_cartpole(output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(USDA_TEMPLATE)
    print(f"Saved cartpole.usda → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output .usda file path")
    args = parser.parse_args()
    create_cartpole(args.output)
