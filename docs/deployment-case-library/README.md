# Finite Element Deployment Case Library

This directory records real forum cases in which a user presents a complex physical or engineering goal through an apparent mesh, convergence, contact, NaN, or solver problem. The research hypothesis is that many such cases are not adequately described as local debugging tasks: the unresolved object is often the finite-element simulation design itself, including physical abstraction, geometry idealization, element and mesh strategy, constitutive model, contact formulation, boundary and loading path, nonlinear solution route, solver capability, and validation plan.

## Current scope

- `calculix-cases.md`: 15 cases from the CalculiX Discourse forum.
- `fenics-cases.md`: 15 cases from the FEniCS Project Discourse forum.
- `schema.md`: annotation schema and preliminary deployment taxonomy.

The initial corpus contains 30 cases collected on 2026-07-31. Each record preserves the original source URL and separates what the user observed from the project's deployment-level interpretation.

## Core distinction

A local debug task assumes that the intended model and correct behavior are already sufficiently specified, and asks why an implementation fails. A deployment-design task still has unresolved choices about what model should exist, how the physical problem should be discretized, which solver route is appropriate, and how numerical success should be distinguished from physical validity.

The corpus therefore treats runtime errors and non-convergence as evidence generated inside a broader design-and-verification loop:

`engineering intent -> modeling assumptions -> candidate FE deployment -> execution -> numerical and physical evidence -> design revision`

## Important caveat

The field `Deployment reading` is a research annotation made for this project. It does not claim that forum participants used the term deployment, nor that every listed issue is exclusively a deployment problem. Some cases also contain ordinary software bugs, solver defects, or unsupported features. The purpose is to preserve the coupled decision structure that is lost when each thread is reduced to a single error message.

## Intended uses

1. Problem-definition evidence for research on automated FE model design, deployment, and verification.
2. Source material for a benchmark whose target is a justified modeling plan rather than a one-line error fix.
3. Retrieval cases for an agent that must distinguish local repair from model-form, discretization, loading-path, or solver-route revision.
4. Evaluation of whether an automated system records assumptions, alternatives, execution evidence, and unresolved uncertainty.
