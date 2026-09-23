# Handover: kinematics background chapter

Context for continuing work on the theory chapter of my master's thesis.
Carried over from a chat session; the current state of the text is in
`kinematics_background.tex`.

## What the chapter is

Background/theory chapter on the kinematics of a ViperX-300, a 6-DOF
open-chain manipulator built entirely from revolute joints. The chapter
builds up from configuration space to forward and inverse kinematics,
so that the kinematic model used later in the thesis follows from it.

Current section order:

1. Configuration Space
2. Degrees of Freedom
3. Anatomy of an Open-Chain Robot
4. Rotation Matrix
5. Homogeneous Transformation Matrix
6. The Denavit-Hartenberg (D-H) Convention
7. Forward and Inverse Kinematics (with Forward / Inverse subsubsections)

Everything is `\subsection` level, so the file is meant to be `\input`
into a chapter, not to stand alone.

## Sources and how they are used

- `lynch2017modern` (Lynch & Park, *Modern Robotics*) — the fundamentals:
  configuration space, degrees of freedom, joint types.
- `siciliano2009robotics` (Siciliano et al., *Robotics: Modelling,
  Planning and Control*) — the rotation matrix section follows his
  development closely: orthonormal frame → rotation matrix → orthogonality
  → elementary rotations → the three geometric meanings → composition and
  current vs. fixed frame. Also the source for the D-H convention.

The rotation matrix section was rewritten to follow Siciliano rather than
Lynch, after comparing both. Keep it that way if it is revised further.

## Conventions to follow

- **Frame notation:** `^A R_B`, `^A T_B`, `^A P_B` — superscript on the left
  is the reference frame, subscript is the frame being described. Siciliano
  uses `R^0_1` for the same thing, so his composition rules carry over
  unchanged. Lynch uses `R_{ab}`, which does *not* match — translate if
  pulling anything from him.
- **Angles:** `\vartheta` (matches the D-H section).
- **Vectors:** bold via `\mathbf{}`, e.g. `\mathbf{q}`, `\mathbf{p}`.
- **Group name:** currently "special orthogonal group" for SO(3). Siciliano
  writes "special orthonormal"; Lynch writes "special orthogonal". Picked
  one deliberately — do not flip it back and forth.
- **Cross-references:** equations are labelled `eq:` and referenced with
  `\eqref{}`.

## Writing style

Prose, not bullet lists. Each concept is motivated before it is defined,
and sections end by pointing forward to the next one (the rotation matrix
section ends on postmultiplication about current frames, which is what the
D-H procedure then uses). Avoid restating a definition twice in adjacent
sections.

## Open items

- **Typos in Configuration Space** (not yet fixed): "collected intro" →
  "into"; "of C-space" → "or C-space"; "it cant only move" → "it can only
  move".
- **Redundancy:** the Homogeneous Transformation Matrix section still
  describes `^A R_B` as "the rotation matrix describing the relative
  orientation", which now repeats the preceding section. Could be shortened
  to just naming the block.
- **Bib keys:** verify `lynch2017modern` and `siciliano2009robotics` match
  the actual entries in the .bib file.
- **Duplicate labels:** the rotation section added `eq:rotation_matrix`,
  `eq:orthogonality`, `eq:elementary_rotations`, `eq:coordinate_transform`,
  `eq:rotation_composition`. Check these do not clash with labels elsewhere
  in the thesis.
- **Figures:** `figures/joints_dof.tex` is referenced. Siciliano-style
  figures for frame rotation and vector representation could be added to
  the rotation section if it needs illustrating.
- **Later chapters:** the D-H section promises a parameter table
  (`tab:dh_params`) and the inverse kinematics section promises an analysis
  of whether the ViperX-300 has a spherical wrist. Both still to be written.
