using SpecialFunctions
using Random

"""
    FBR(a, b, l, u)

Four-parameter beta distribution representation for LCS rules.
- a, b: Shape parameters controlling distribution form
- l, u: Support interval [l, u] defining feature subspace
- beta_val: Precomputed beta function value for efficiency
"""
mutable struct FBR
    a::Float64
    b::Float64
    l::Float64
    u::Float64
    beta_val::Float64
end

function FBR(a, b, l, u)
    # Precompute beta function value to avoid repeated calculations
    beta_val = beta(a, b)
    return FBR(a, b, l, u, beta_val)
end

"""
    get_mode(fbr::FBR)::Float64

Calculate mode of beta distribution.
Modes define peak locations for fuzzy rules:
- Symmetric bell: α,β > 1
- Monotonic: α=1 or β=1 (left/right triangular)
- Crisp interval: α=β=1 (rectangular)
"""
@inline function get_mode(fbr::FBR)::Float64
    # Parameter validation
    if !(fbr.a >= 1.0 && fbr.b >= 1.0 && fbr.l <= fbr.u)
        error("Not ordered: $(fbr.a), $(fbr.b), $(fbr.l), $(fbr.u)")
    end

    if fbr.a > 1.0 && fbr.b > 1.0 # Symmetric/asymmetric bell
        return fbr.l + ((fbr.a - 1)/(fbr.a + fbr.b - 2)) * (fbr.u - fbr.l)
    elseif fbr.a == 1.0 && fbr.b > 1.0 # Left triangular (monotonic)
        return fbr.l
    elseif fbr.a > 1.0 && fbr.b == 1.0 # Right triangular (monotonic)
        return fbr.u
    else
        error("Mode undefined for crisp intervals (α=β=1)")
    end
end

"""
    is_bell(fbr::FBR)::Bool

Check if rule represents fuzzy form (non-rectangular).
Used in subsumption condition checks.
"""
function is_bell(fbr::FBR)::Bool
    if (fbr.a > 1.0 && fbr.b > 1.0) || (fbr.a == 1.0 && fbr.b > 1.0) || (fbr.a > 1.0 && fbr.b == 1.0)
        return true
    else
        return false
    end
end

"""
    is_rectangle(fbr::FBR)::Bool

Check if rule represents crisp interval (α=β=1).
"""
function is_rectangle(fbr::FBR)::Bool
    if fbr.a == 1.0 && fbr.b == 1.0
        return true
    else
        return false
    end
end

"""
    get_kurtosis(fbr::FBR)::Float64

Calculate distribution kurtosis for subsumption condition.
Lower kurtosis indicates broader peaks for generalization.
"""
@inline function get_kurtosis(fbr::FBR)::Float64
    numerator::Float64 = 3.0 * (fbr.a + fbr.b + 1.0) * (2.0 * (fbr.a + fbr.b) ^ 2.0 + fbr.a * fbr.b * (fbr.a + fbr.b - 6.0))
    denominator::Float64 = fbr.a * fbr.b * (fbr.a + fbr.b + 2.0) * (fbr.a + fbr.b + 3.0)

    return numerator / denominator
end

"""
    get_pdf(fbr::FBR, x::Float64)::Float64

Calculate probability density function value at x.
Returns 0 outside [l,u] interval.
"""
@inline function get_pdf(fbr::FBR, x::Float64)::Float64
    if !(fbr.l <= x <= fbr.u)
        return 0.0
    else
        numerator::Float64 = (x - fbr.l)^(fbr.a - 1) * (fbr.u - x)^(fbr.b - 1)
        denominator::Float64 = fbr.beta_val * (fbr.u - fbr.l)^(fbr.a + fbr.b - 1)
        return numerator / denominator
    end
end

"""
    get_membership_value(fbr::FBR, x)::Float64

Calculate normalized membership degree [0,1] for input x.
Handles missing values ("?") and crisp intervals.
"""
@inline function get_membership_value(fbr::FBR, x::Union{Float64, String})::Float64
    if !(fbr.a >= 1.0 && fbr.b >= 1.0 && fbr.l <= fbr.u)
        error("Not ordered: $(fbr.a), $(fbr.b), $(fbr.l), $(fbr.u)")
    end

    # Handle missing values
    if x == "?"
        return 1.0
    end

    if !(fbr.l <= x <= fbr.u)
        return 0.0
    end

     # Crisp interval fast path (no PDF calculation needed)
    if is_rectangle(fbr)
        return 1.0
    end

    # Calculate and normalize PDF value
    state_pdf_value::Float64 = get_pdf(fbr, x)
    mode::Float64 = get_mode(fbr)
    mode_pdf_value::Float64 = get_pdf(fbr, mode)

    # Normalize to [0,1] range
    return state_pdf_value / mode_pdf_value
end

"""
    ordered_condition!(fbr::FBR)

Ensure l ≤ u and handle edge cases for numerical stability.
Prevents matching degree calculation errors.
"""
@inline function ordered_condition!(fbr::FBR)
    if fbr.l > fbr.u # Fix inverted interval
        fbr.l, fbr.u = fbr.u, fbr.l
    end

    if fbr.l == fbr.u # Prevent zero-width intervals
        fbr.l = max(0, fbr.l - 0.025)
        fbr.u = min(1, fbr.u + 0.025)
    end
end

"""
    is_equal(f1::FBR, f2::FBR)::Bool

Compare two FBR instances for equality.
Used in rule similarity checks during subsumption.
"""
function is_equal(f1::FBR, f2::FBR)::Bool
    if f1.a == f2.a && f1.b == f2.b && f1.l == f2.l && f1.u == f2.u
        return true
    else
        return false
    end
end