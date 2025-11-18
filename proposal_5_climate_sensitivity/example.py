"""
Quick Example: Multi-Constraint Framework for Climate Sensitivity

This script demonstrates basic usage of the framework.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import modules
from src.paleoclimate import LGMConstraint, mPWPConstraint
from src.observational import HistoricalWarmingConstraint
from src.process_based import CloudFeedbackConstraint
from src.integration import MultiConstraintFramework


def main():
    """Run example ECS estimation"""

    print("=" * 70)
    print("Multi-Constraint Framework for Climate Sensitivity")
    print("=" * 70)

    # Step 1: Initialize constraints
    print("\n1. Initializing constraints...")
    lgm = LGMConstraint()
    mpwp = mPWPConstraint()
    historical = HistoricalWarmingConstraint()
    cloud = CloudFeedbackConstraint()

    # Step 2: Calculate individual estimates
    print("\n2. Calculating individual constraint estimates...")
    constraints = [lgm, mpwp, historical, cloud]

    print("\nIndividual Constraint Estimates:")
    print("-" * 70)
    for constraint in constraints:
        result = constraint.calculate_constraint()
        print(f"{result['name']:12s} | Median: {result['median']:.2f} K | "
              f"90% CI: [{result['ci_90'][0]:.2f}, {result['ci_90'][1]:.2f}] K")

    # Step 3: Integrate constraints
    print("\n3. Integrating constraints using Bayesian framework...")
    mcf = MultiConstraintFramework(
        constraints=constraints,
        prior='uniform',
        ecs_range=(1.0, 6.0),
        test_independence=True
    )

    posterior = mcf.integrate_constraints(method='bayesian')

    # Step 4: Display results
    print("\n" + "=" * 70)
    print("COMBINED ECS ESTIMATE")
    print("=" * 70)
    print(f"\nMedian ECS:        {posterior.median():.2f} K")
    print(f"Mean ECS:          {posterior.mean():.2f} K")
    print(f"Mode ECS:          {posterior.mode():.2f} K")
    print(f"\n66% CI:            [{posterior.credible_interval(0.66)[0]:.2f}, "
          f"{posterior.credible_interval(0.66)[1]:.2f}] K")
    print(f"90% CI:            [{posterior.credible_interval(0.90)[0]:.2f}, "
          f"{posterior.credible_interval(0.90)[1]:.2f}] K")

    # Step 5: Check independence
    if mcf.independence_results:
        print("\n" + "=" * 70)
        print("CONSTRAINT INDEPENDENCE TESTS")
        print("=" * 70)
        for pair, results in mcf.independence_results.items():
            print(f"\n{pair}:")
            print(f"  Independent: {results['independent']}")
            print(f"  {results['interpretation']}")

    # Step 6: Plot results
    print("\n4. Generating plot...")
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        posterior.plot(ax=ax)
        plt.savefig('proposal_5_climate_sensitivity/results/ecs_posterior.png',
                    dpi=300, bbox_inches='tight')
        print("   Plot saved to: results/ecs_posterior.png")
    except Exception as e:
        print(f"   Note: Could not save plot ({e})")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
