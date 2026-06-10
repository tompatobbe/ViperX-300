#!/usr/bin/env python3
"""
phi_to_urdf.py — Convert identified phi.npy to a URDF with identified inertial parameters
===========================================================================================

Two modes:

  1. Standalone — generate a complete minimal URDF from the DH params + phi
     (no mesh files, correct inertials and joint structure):

       python phi_to_urdf.py phi_fast.npy -o viper300_sysid.urdf

  2. Template — patch the <inertial> blocks of an existing ViperX-300 URDF
     while keeping all meshes, visuals, and collision geometry intact:

       python phi_to_urdf.py phi_fast.npy --template original.urdf -o patched.urdf

     Optionally override which link names in the URDF correspond to phi links 1-6:
       --link-names link1_name link2_name link3_name link4_name link5_name link6_name

     Default link names (Interbotix ViperX-300):
       waist_link upper_arm_link forearm_link wrist_link gripper_link ee_arm

Phi format (78 floats, 13 per link × 6 links in proximal→distal order):
  [m, m·cx, m·cy, m·cz, Jxx, Jxy, Jxz, Jyy, Jyz, Jzz, Fv, Fc, F0]

Conversions performed:
  - CoM:  c = mc / m                          (first mass moments → CoM position)
  - J_CoM = J_O − m·(‖c‖²I − c·cᵀ)          (parallel axis: origin → CoM)
  - Fv → URDF <dynamics damping=...>
  - Fc → URDF <dynamics friction=...>
  - F0 is not representable in URDF (offset torque, keep for reference in comments)

Frame note (template mode):
  The identified phi is expressed in the DH link frames (Craig convention).
  The URDF link frames must coincide with the DH frames for the inertia tensors
  to transfer directly.  Interbotix's ViperX-300 URDF uses matching frames for
  links 1-6.  If you use a custom URDF with different frame orientations you
  will need to rotate J_CoM and c by the difference rotation.
"""

import numpy as np
import xml.etree.ElementTree as ET
import argparse
import os
import sys

import pipeline_artifacts

PIPELINE_NAME    = "phi_to_urdf"
PIPELINE_VERSION = "1.0"   # bump when URDF generation logic changes

# =============================================================================
# DH parameters (must match sysid_fast.py / sysid_subsample.py)
# =============================================================================

L1 = 0.12675
L2 = 0.30594
L3 = 0.21981
L4 = 0.08021
L6 = 0.13658   # end-effector offset (used for ee joint, NOT as d_6)

DH_PARAMS = np.array([
    # alpha_prev   a_prev    d_i          theta_offset
    [0.0,          0.0,      L1,           0.0          ],
    [3*np.pi/2,    0.0,      0.0,         -0.437*np.pi  ],
    [0.0,          L2,       0.0,         -0.063*np.pi  ],
    [3*np.pi/2,    0.0,      L3+L4,        0.0          ],
    [  np.pi/2,    0.0,      0.0,          0.0          ],
    [3*np.pi/2,    0.0,      0.0,          0.0          ],
], dtype=float)

N_JOINTS   = 6
N_PARAMS   = 13

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
LINK_NAMES  = ["waist_link", "upper_arm_link", "forearm_link",
               "wrist_link", "gripper_link", "ee_arm"]

# ViperX-300 joint limits: [lower_rad, upper_rad, effort_Nm, velocity_rad_s]
JOINT_LIMITS = [
    (-3.14159, 3.14159,  4.1,  3.14159),   # waist
    (-1.8763,  1.9897,  10.6,  2.6105 ),   # shoulder
    (-2.1521,  1.5708,  10.6,  2.6105 ),   # elbow
    (-3.14159, 3.14159,  4.1,  3.14159),   # forearm_roll
    (-1.7453,  2.1468,   4.1,  3.14159),   # wrist_angle
    (-3.14159, 3.14159,  4.1,  3.14159),   # wrist_rotate
]


# =============================================================================
# DH kinematics helpers
# =============================================================================

def _dh_transform(alpha, a, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ])


def _relative_transforms_at_zero():
    """Return T_{i-1,i}(q=0) for each of the 6 joints (relative, not accumulated)."""
    return [_dh_transform(alpha, a, d, theta_off)
            for alpha, a, d, theta_off in DH_PARAMS]


def _mat_to_rpy(R):
    """
    Rotation matrix → URDF RPY angles [roll, pitch, yaw].
    URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (extrinsic x-y-z).
    """
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    cp = np.cos(pitch)
    if abs(cp) > 1e-9:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw  = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    else:
        # Gimbal lock: pitch ≈ ±90°
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw  = 0.0
    return roll, pitch, yaw


# =============================================================================
# Inertia / CoM conversions
# =============================================================================

def _j_origin_to_jcom(J_O, m, mc):
    """
    Parallel axis theorem: shift inertia tensor from link origin to CoM.
      J_CoM = J_O − m · (‖c‖²·I − c·cᵀ)
    where c = mc/m is the CoM position vector.
    """
    c = mc / m
    return J_O - m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))


# Validation-only escape hatch: when an *unregularised* SDP solution has
# collapsed (near-zero) link masses, set this > 0 to clamp the mass used as the
# CoM/parallel-axis divisor. Such links also have near-zero first moment mc, so
# clamping only the divisor preserves mc (= m·c) and J_O exactly — and those are
# precisely what RNEA inverse-dynamics torque depends on. The emitted URDF is
# therefore torque-faithful (good for sanity-validation) but its per-link masses
# are NOT a physical realisation; use the entropic regulariser for a real URDF.
_MASS_FLOOR = 0.0


def _parse_link(phi, i):
    """
    Extract processed inertial parameters for link i from the full phi vector.
    Returns: (m, c_com, J_com, Fv, Fc, F0)
      c_com  : CoM position in DH link frame [m]
      J_com  : 3×3 inertia at CoM in DH link frame [kg·m²]
    """
    base = i * N_PARAMS
    p    = phi[base : base + N_PARAMS]
    m    = float(p[0])
    mc   = p[1:4].astype(float)
    J_O  = np.array([[p[4], p[5], p[6]],
                     [p[5], p[7], p[8]],
                     [p[6], p[8], p[9]]], dtype=float)
    Fv   = float(p[10])
    Fc   = float(p[11])
    F0   = float(p[12])

    if abs(m) < 1e-9:
        if _MASS_FLOOR > 0.0:
            m = _MASS_FLOOR        # clamp divisor only; mc & J_O preserved → torque-faithful
        else:
            raise ValueError(f"Link {i+1}: near-zero mass m={m:.2e} — phi may be invalid")

    c_com = mc / m
    J_com = _j_origin_to_jcom(J_O, m, mc)

    return m, c_com, J_com, Fv, Fc, F0


# =============================================================================
# XML helpers
# =============================================================================

def _inertial_element(m, c, J):
    """Build an <inertial> Element from mass, CoM position, and J_CoM."""
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin",
                  xyz=f"{c[0]:.8f} {c[1]:.8f} {c[2]:.8f}",
                  rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{m:.8f}")
    ET.SubElement(inertial, "inertia",
                  ixx=f"{J[0,0]:.10f}", ixy=f"{J[0,1]:.10f}", ixz=f"{J[0,2]:.10f}",
                  iyy=f"{J[1,1]:.10f}", iyz=f"{J[1,2]:.10f}", izz=f"{J[2,2]:.10f}")
    return inertial


def _indent(elem, level=0):
    """Add pretty-print indentation in-place (works on Python < 3.9)."""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


# =============================================================================
# Mode 1: Generate standalone minimal URDF
# =============================================================================

def generate_standalone(phi, robot_name="viper300_sysid"):
    """
    Build a complete URDF from DH params + identified phi.
    No mesh files — suitable for simulation in Gazebo/MuJoCo or control use.

    Joint origins are computed numerically from DH transforms at q=0.
    The joint axis is [0 0 1] in the joint frame (Craig DH: rotation around z_{i-1}).
    """
    T_rel = _relative_transforms_at_zero()   # list of 6 relative (4×4) transforms

    root = ET.Element("robot", name=robot_name)

    # Base link (world anchor)
    base_link = ET.SubElement(root, "link", name="base_link")
    base_inertial = ET.SubElement(base_link, "inertial")
    ET.SubElement(base_inertial, "mass", value="0.0")
    ET.SubElement(base_inertial, "inertia",
                  ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")

    for i in range(N_JOINTS):
        parent_name = "base_link" if i == 0 else LINK_NAMES[i - 1]
        link_name   = LINK_NAMES[i]
        joint_name  = JOINT_NAMES[i]
        lo, hi, eff, vel = JOINT_LIMITS[i]
        m, c, J, Fv, Fc, F0 = _parse_link(phi, i)

        # Joint origin from DH transform at q=0 (relative to parent)
        T  = T_rel[i]
        xyz = T[:3, 3]
        rpy = _mat_to_rpy(T[:3, :3])

        joint = ET.SubElement(root, "joint", name=joint_name, type="revolute")
        ET.SubElement(joint, "origin",
                      xyz=f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}",
                      rpy=f"{rpy[0]:.8f} {rpy[1]:.8f} {rpy[2]:.8f}")
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child",  link=link_name)
        ET.SubElement(joint, "axis",   xyz="0 0 1")
        ET.SubElement(joint, "limit",
                      lower=f"{lo:.6f}", upper=f"{hi:.6f}",
                      effort=f"{eff:.3f}", velocity=f"{vel:.6f}")
        ET.SubElement(joint, "dynamics",
                      damping=f"{max(Fv, 0.0):.8f}",
                      friction=f"{max(Fc, 0.0):.8f}")

        # Link with inertial block
        link = ET.SubElement(root, "link", name=link_name)
        link.append(_inertial_element(m, c, J))
        link.append(_comment_f0(F0, i))

    # Fixed end-effector joint (L6 offset along x of last DH frame)
    ee_joint = ET.SubElement(root, "joint", name="ee_arm", type="fixed")
    ET.SubElement(ee_joint, "origin",
                  xyz=f"{L6:.8f} 0.0 0.0", rpy="0 0 0")
    ET.SubElement(ee_joint, "parent", link=LINK_NAMES[-1])
    ET.SubElement(ee_joint, "child",  link="ee_link")
    ET.SubElement(root, "link", name="ee_link")

    return root


def _comment_f0(F0, i):
    """Return an XML comment with the F0 offset torque (not representable in URDF)."""
    c = ET.Comment(f" Identified torque offset F0[{i+1}] = {F0:.6f} Nm "
                   f"(feed-forward term, add manually to your controller) ")
    return c


# =============================================================================
# Mode 2: Patch an existing URDF template
# =============================================================================

def patch_template(phi, template_path, link_names):
    """
    Parse an existing URDF and replace each link's <inertial> block with
    values from the identified phi vector.

    Also updates <dynamics damping friction> on the matching joints.
    """
    tree = ET.parse(template_path)
    root = tree.getroot()

    for i, lname in enumerate(link_names):
        # Find the link element
        link_el = root.find(f".//link[@name='{lname}']")
        if link_el is None:
            print(f"[warn] Link '{lname}' not found in template — skipping", file=sys.stderr)
            continue

        m, c, J, Fv, Fc, F0 = _parse_link(phi, i)

        # Remove old inertial block (if any)
        old = link_el.find("inertial")
        if old is not None:
            link_el.remove(old)

        # Insert new inertial block
        link_el.insert(0, _inertial_element(m, c, J))
        link_el.insert(1, _comment_f0(F0, i))

        # Update joint dynamics (find joint whose child = this link)
        joint_el = root.find(f".//joint[child[@link='{lname}']]")
        if joint_el is not None:
            dyn = joint_el.find("dynamics")
            if dyn is None:
                dyn = ET.SubElement(joint_el, "dynamics")
            dyn.set("damping",  f"{max(Fv, 0.0):.8f}")
            dyn.set("friction", f"{max(Fc, 0.0):.8f}")
            print(f"  Link {i+1} '{lname}': updated inertial + dynamics")
        else:
            print(f"  Link {i+1} '{lname}': updated inertial (joint not found)")

    return root


# =============================================================================
# Summary printout
# =============================================================================

def print_summary(phi):
    print("\nIdentified inertial parameters:")
    print(f"{'Link':<14} {'m [kg]':>9} {'cx [m]':>9} {'cy [m]':>9} {'cz [m]':>9} "
          f"{'Fv [Nm·s]':>11} {'Fc [Nm]':>9} {'F0 [Nm]':>9}")
    print("-" * 90)
    for i in range(N_JOINTS):
        m, c, J, Fv, Fc, F0 = _parse_link(phi, i)
        print(f"  {LINK_NAMES[i]:<12} {m:>9.4f} {c[0]:>9.5f} {c[1]:>9.5f} {c[2]:>9.5f} "
              f"{Fv:>11.5f} {Fc:>9.5f} {F0:>9.5f}")

    print("\nInertia at CoM [kg·m²] (diagonal):")
    print(f"  {'Link':<12} {'Ixx':>12} {'Iyy':>12} {'Izz':>12}")
    for i in range(N_JOINTS):
        m, c, J, *_ = _parse_link(phi, i)
        print(f"  {LINK_NAMES[i]:<12} {J[0,0]:>12.6f} {J[1,1]:>12.6f} {J[2,2]:>12.6f}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert identified phi.npy to a URDF with updated inertial parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument("phi",
                        help="Path to identified parameter vector (.npy), shape (78,)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write directly to this path, bypassing the artifact system. "
                             "Prompts before overwriting. Omit to use managed outputs/.")
    parser.add_argument("--template",
                        help="Existing URDF to patch (keeps meshes/visuals/collisions). "
                             "If omitted, a minimal standalone URDF is generated.")
    parser.add_argument("--link-names", nargs=6, metavar="LINK",
                        default=LINK_NAMES,
                        help="6 URDF link names in proximal→distal order "
                             f"(default: {' '.join(LINK_NAMES)})")
    parser.add_argument("--robot-name", default="viper300_sysid",
                        help="Robot name in the generated URDF (standalone mode only)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Suppress parameter table printout")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if an identical artifact already exists")
    parser.add_argument("--outputs-dir", default=None,
                        help="Override output root directory (default: outputs/)")
    parser.add_argument("--mass-floor", type=float, default=0.0,
                        help="Validation-only: clamp near-zero link masses to this "
                             "value [kg] instead of erroring. Preserves mc & J_O "
                             "(torque-faithful) but the URDF masses are NOT a "
                             "physical realisation — use the entropic regulariser "
                             "(sysid --entropic) for a real URDF.")
    args = parser.parse_args()

    if args.mass_floor > 0.0:
        global _MASS_FLOOR
        _MASS_FLOOR = args.mass_floor
        print(f"[mass-floor] near-zero masses clamped to {args.mass_floor:g} kg "
              f"(torque-faithful; masses are not a physical realisation)")

    # Load phi
    phi = np.load(args.phi)
    if phi.shape != (78,):
        sys.exit(f"Error: expected phi shape (78,), got {phi.shape}")

    if not args.no_summary:
        print(f"Loaded phi from: {args.phi}")
        print_summary(phi)

    # -o path: legacy direct-write mode, no artifact management.
    if args.output is not None:
        if os.path.exists(args.output):
            ans = input(f"'{args.output}' already exists. Overwrite? [y/N] ").strip().lower()
            if ans != "y":
                sys.exit("Aborted.")
        _write_urdf(phi, args)
        print(f"\nWrote: {args.output}")
        return

    # Artifact-managed mode.
    config = {
        "mode":       "template" if args.template else "standalone",
        "template":   str(args.template) if args.template else None,
        "link_names": list(args.link_names),
        "robot_name": args.robot_name,
        "mass_floor": args.mass_floor,
    }

    # Cache-hit check.
    if not args.force:
        art_path, _ = pipeline_artifacts.artifact_path(
            args.phi, PIPELINE_NAME, PIPELINE_VERSION, config,
            args.outputs_dir, suffix=".urdf",
        )
        if art_path.exists():
            _, json_path = pipeline_artifacts.artifact_path(
                args.phi, PIPELINE_NAME, PIPELINE_VERSION, config,
                args.outputs_dir, suffix=".urdf",
            )
            print(f"\n[cache] Artifact already exists — skipping generation.")
            print(f"  {art_path}")
            print("  Pass --force to regenerate.")
            return

    # Generate URDF content as a string.
    content = _build_urdf_string(phi, args)

    urdf_path, json_path = pipeline_artifacts.save_text_artifact(
        content, args.phi, PIPELINE_NAME, PIPELINE_VERSION, config,
        suffix=".urdf", outputs_root=args.outputs_dir,
        allow_overwrite=args.force,
    )
    print(f"\nWrote   →  {urdf_path}")
    print(f"Sidecar →  {json_path}")


def _build_urdf_string(phi: np.ndarray, args) -> str:
    """Generate the URDF XML and return it as a string."""
    if args.template:
        print(f"\nPatching template: {args.template}")
        root = patch_template(phi, args.template, args.link_names)
    else:
        print(f"\nGenerating standalone URDF (robot name: '{args.robot_name}')...")
        root = generate_standalone(phi, robot_name=args.robot_name)

    # Provenance comment — no timestamp so content is deterministic.
    root.insert(0, ET.Comment(
        f" Generated by phi_to_urdf {PIPELINE_VERSION} | phi: {args.phi} "
    ))

    _indent(root)
    import io as _io
    buf = _io.StringIO()
    buf.write('<?xml version="1.0" ?>\n')
    ET.ElementTree(root).write(buf, encoding="unicode")
    return buf.getvalue()


def _write_urdf(phi: np.ndarray, args) -> None:
    """Direct write to args.output (legacy -o mode)."""
    if args.template:
        root = patch_template(phi, args.template, args.link_names)
    else:
        root = generate_standalone(phi, robot_name=args.robot_name)
    root.insert(0, ET.Comment(
        f" Generated by phi_to_urdf {PIPELINE_VERSION} | phi: {args.phi} "
    ))
    _indent(root)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" ?>\n')
        ET.ElementTree(root).write(f, encoding="unicode")


if __name__ == "__main__":
    main()
