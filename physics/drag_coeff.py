import math


def calculate_drag_coefficient(
    reynolds_number: float, aspect_ratio: float, incident_angle_deg: float
) -> float:
    """
    Args:
        reynolds_number (float)
        aspect_ratio (float)
        incident_angle_deg (float)
    Returns:
        float: The calculated drag coefficient (Cd).
    """

    c1, c2 = 18.7371, 0.2883
    c3, c4 = 7.9738, -0.5126
    c5, c6 = 0.1938, -1.1848
    c7, c8, c9, c10 = -0.5531, 2.6334, 0.2199, 0.9865

    Re = reynolds_number
    Ar = aspect_ratio

    # Convert incident angle from degrees to radians for math.sin()
    zeta = math.radians(incident_angle_deg)

    # Break down the formula into its constituent parts for clarity
    term1 = (c1 / Re) * (Ar**c2)
    term2 = (c3 / math.sqrt(Re)) * (Ar**c4)
    term3 = c5 * (Ar**c6)

    # The last term accounts for the incident angle's effect
    term4_part1 = (Ar**c7) * (Ar - 1)
    term4_part2 = c8 / (Re**c9)
    term4_part3 = math.sin(c10 * zeta) ** 2
    term4 = term4_part1 * term4_part2 * term4_part3

    # Sum the terms to get the final drag coefficient
    Cd = term1 + term2 + term3 + term4

    return Cd


def main():
    test_cases = [
        {"Re": 10, "Ar": 0.5, "angle": 0.0},
        {
            "Ar": 0.5000017768899694,
            "angle": 0.000148118675391,
            "Re": 10.000065199649375,
        },
    ]

    for case in test_cases:
        cd = calculate_drag_coefficient(case["Re"], case["Ar"], case["angle"])
        print(f"Re={case['Re']}, Ar={case['Ar']}, angle={case['angle']} → Cd={cd:.6f}")


if __name__ == "__main__":
    main()
