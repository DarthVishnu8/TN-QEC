using ITensors
using Plots

# Define the site indices for 7 qubits
N = 7
s = siteinds("S=1/2", N)

# Existing operator and initialization functions from the previous implementation
# (Copying over the functions from the original code)
function ITensors.op(::OpName"Z", ::SiteType"S=1/2", s::Index)
    mat = [1 0
           0 -1]
    return ITensor(mat, s', s)
end

function ITensors.op(::OpName"X", ::SiteType"S=1/2", s::Index)
    mat = [0 1
           1 0]
    return ITensor(mat, s', s)
end

function ITensors.op(::OpName"H", ::SiteType"S=1/2", s::Index)
    mat = [1 1
           1 -1]/sqrt(2)
    return ITensor(mat, s', s)
end

function initialize_state(α::Number, β::Number)
    ψ = MPS(s)
    
    # Set first qubit to α|0⟩ + β|1⟩
    ψ[1] = ITensor([α, β], s[1])
    
    # Set other qubits to |0⟩
    for j in 2:N
        ψ[j] = ITensor([1.0, 0.0], s[j])
    end
    
    return normalize!(ψ)
end

function encode_steane(ψ::MPS)
    encoding_ops = [
        ("H", 5), ("H", 6), ("H", 7),  # Create |+⟩ states
        ("CX", 1, 2), ("CX", 1, 3), # first round
        ("CX", 7, 1), ("CX", 7, 2), ("CX", 7, 4), # second round
        ("CX", 6, 1), ("CX", 6, 3), ("CX", 6, 4), # third round
        ("CX", 5, 2), ("CX", 5, 3), ("CX", 5, 4) # fourth round
    ]
    ψ_encoded = apply(ops(encoding_ops, s), ψ)
    return normalize!(ψ_encoded)
end

# Updated compute_syndrome_probabilities function
function compute_syndrome_probabilities(ψ::MPS)
    probs = Dict{Tuple{Int,Int,Int,Int,Int,Int}, Float64}()
    
    # Define the stabilizer operators and their locations
    stabilizers = [
        ("X", [4,5,6,7]),  # X4567
        ("X", [2,3,6,7]),  # X2367
        ("X", [1,3,5,7]),  # X1357
        ("Z", [4,5,6,7]),  # Z4567
        ("Z", [2,3,6,7]),  # Z2367
        ("Z", [1,3,5,7])   # Z1357
    ]
    
    for j in 1:length(ψ)
        ψ[j] = noprime(ψ[j])
    end
    
    # Iterate through all possible syndrome patterns (64 combinations)
    for s1 in 0:1, s2 in 0:1, s3 in 0:1, s4 in 0:1, s5 in 0:1, s6 in 0:1
        syndromes = [s1, s2, s3, s4, s5, s6]
        
        # Make a copy of the state to work with
        ψ_projected = copy(ψ)
        
        # Apply each stabilizer projector
        for (idx, (op_type, qubits)) in enumerate(stabilizers)
            sign = (-1)^syndromes[idx]
            # Store original state before applying stabilizer
            ψ_orig = copy(ψ_projected)
            
            # Apply operator to each qubit in the stabilizer
            for q in qubits
                ψ_projected = orthogonalize!(ψ_projected, q)
                ψ_projected[q] = op(op_type, s[q]) * ψ_projected[q]
            end
            
            # Unprime all tensors before addition
            for j in 1:length(ψ_projected)
                ψ_projected[j] = noprime(ψ_projected[j])
            end
            for j in 1:length(ψ_orig)
                ψ_orig[j] = noprime(ψ_orig[j])
            end
            
            # Form the projector (I ± S)/2 where S is the stabilizer
            ψ_projected = (ψ_orig + sign * ψ_projected)/2.0
        end
        
        # Compute probability
        prob = abs2(inner(ψ', ψ_projected))
        if prob > 1e-10  # Only store non-negligible probabilities
            probs[(s1,s2,s3,s4,s5,s6)] = prob
        end
    end
    
    # Normalize probabilities
    total = sum(values(probs))
    for k in keys(probs)
        probs[k] /= total
    end
    
    return probs
end

# Apply amplitude damping error
function apply_amplitude_damping(ψ::MPS, site::Int, γ::Float64)
    ψ = orthogonalize!(ψ, site)
    T = ψ[site]
    si = siteind(ψ, site)
    
    K0 = ITensor([1.0 0.0; 0.0 sqrt(1-γ)], si', si)
    K1 = ITensor([0.0 sqrt(γ); 0.0 0.0], si', si)
    
    ψ0 = copy(ψ)
    ψ0[site] = K0 * T
    normalize!(ψ0)
    
    ψ1 = copy(ψ)
    ψ1[site] = K1 * T
    normalize!(ψ1)
    
    p0 = real(inner(ψ0', ψ0))
    p1 = real(inner(ψ1', ψ1))
    
    total_prob = p0 + p1
    p0 /= total_prob
    p1 /= total_prob
    
    if rand() < p0
        return normalize!(ψ0)
    else
        return normalize!(ψ1)
    end
end

# Comprehensive test function
function test_steane_code_full_analysis(α::Number, β::Number, γ::Float64)
    # Initialize and encode
    ψ_initial = initialize_state(α, β)
    println("Initial state prepared")

    ψ_encoded = encode_steane(ψ_initial)
    println("State encoded using Steane code")

    # Apply amplitude damping to all qubits
    ψ_with_error = copy(ψ_encoded)
    for error_site in 1:7
        ψ_with_error = apply_amplitude_damping(ψ_with_error, error_site, γ)
    end
    println("\nApplied amplitude damping error with γ = $γ on all qubits")

    # Compute syndrome probabilities
    probs = compute_syndrome_probabilities(ψ_with_error)
    
    return ψ_with_error, probs
end

# Run test with parameters α, β, γ  
ψ_final, syndrome_probs = test_steane_code_full_analysis(0.7, 0.7, 0.5)

# Optional: Display non-negligible syndrome probabilities
for (syndrome, prob) in sort(collect(syndrome_probs), by=x->x[2], rev=true)
    if prob > 0.001
        println("P$(syndrome) = $(round(prob, digits=4))")
    end
end
