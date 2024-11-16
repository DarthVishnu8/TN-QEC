using ITensors

# Define the site indices
N = 5  # Logical + Ancilla + Measurement qubits
s = siteinds("S=1/2", N)

# Add Z operator definition
function ITensors.op(::OpName"Z", ::SiteType"S=1/2", s::Index)
    mat = [1 0
           0 -1]
    return ITensor(mat, s', s)
end

function encode_state(ψ::MPS)
    encoding_ops = [
        ("CX", 1, 2),
        ("CX", 1, 3)
    ]
    ψ_encoded = apply(ops(encoding_ops, s), ψ)
    return normalize!(ψ_encoded)
end

# Function to apply amplitude damping error on a specific qubit
function apply_amplitude_damping(ψ::MPS, site::Int, γ::Float64)
    # Orthogonalize to the site we're operating on
    # ψ = orthogonalize!(ψ, site)
    
    # Extract the tensor at the target site
    T = ψ[site]
    si = siteind(ψ, site)
    
    # Create Kraus operators with correct index structure
    K0 = ITensor([1.0 0.0; 0.0 sqrt(1-γ)], si', si)
    K1 = ITensor([0.0 sqrt(γ); 0.0 0.0], si', si)
    
    # Apply K0 and store result
    ψ0 = copy(ψ)
    ψ0[site] = K0 * T
    normalize!(ψ0)
    
    # Apply K1 and store result
    ψ1 = copy(ψ)
    ψ1[site] = K1 * T
    normalize!(ψ1)
    
    # Calculate probabilities
    p0 = real(inner(ψ0', ψ0))
    p1 = real(inner(ψ1', ψ1))
    
    # Normalize probabilities
    total_prob = p0 + p1
    p0 /= total_prob
    p1 /= total_prob
    
    # Randomly choose outcome based on probabilities
    if rand() < p0
        return normalize!(ψ0)
    else
        return normalize!(ψ1)
    end
end

# Function to initialize an arbitrary qubit state |ψ> = α|0⟩ + β|1⟩
function initialize_state(α::Number, β::Number)
    ψ0 = MPS(s)
    
    # Define the initial state on the logical qubit
    ψ0[1] = ITensor([α, β], s[1])
    
    # Initialize remaining qubits in |0⟩ state
    for i in 2:N
        ψ0[i] = ITensor([1.0, 0.0], s[i])
    end
    return ψ0
end


# Function to compute expectation value of ZZ operator
function expectation_ZZ(ψ::MPS, qubit1::Int, qubit2::Int)
    # Orthogonalize to first qubit
    ψ = orthogonalize!(copy(ψ), qubit1)
    
    # Get Z operators
    Z1 = op("Z", s[qubit1])
    Z2 = op("Z", s[qubit2])
    
    # Apply Z1
    T1 = ψ[qubit1]
    T1 = Z1 * T1
    ψ_temp = copy(ψ)
    ψ_temp[qubit1] = T1
    
    # Apply Z2
    ψ_temp = orthogonalize!(ψ_temp, qubit2)
    T2 = ψ_temp[qubit2]
    T2 = Z2 * T2
    ψ_temp[qubit2] = T2
    
    # Compute expectation value
    return real(inner(ψ, ψ_temp))
end

function display_mps_state(ψ::MPS)
    N = length(ψ)
    println("Final state coefficients in computational basis:")
    # Iterate over all 2^N basis states
    for i in 0:(2^N - 1)
        # Generate the binary representation for the basis state
        bitstring = string(i, base=2, pad=N)  # Use padding to get N bits
        # Convert the bitstring to basis states (↑ for 0, ↓ for 1)
        state = [c == '0' ? "↑" : "↓" for c in bitstring]
       
        # Calculate the coefficient for this basis state
        coeff = inner(ψ, MPS(s, state))
       
        # Print the basis state and its coefficient if non-zero
        if abs(coeff) > 1e-10
            println("|", bitstring, "⟩: ", coeff)
        end
    end
end

# Function to compute syndrome probabilities
function compute_syndrome_probabilities(ψ::MPS)
    # Compute ZZ expectation values
    Z₁Z₂ = expectation_ZZ(ψ, 1, 2)
    Z₁Z₃ = expectation_ZZ(ψ, 1, 3)
    
    # Initialize probability dictionary
    probs = Dict{Tuple{Int,Int}, Float64}()
    
    # Compute probabilities for each syndrome pattern using stabilizer projectors
    # P(s₁s₂) = ⟨ψ|(I + (-1)^s₁Z₁Z₂)(I + (-1)^s₂Z₁Z₃)|ψ⟩/4
    
    # For s₁s₂ = 00
    probs[(0,0)] = (1 + Z₁Z₂ + Z₁Z₃ + Z₁Z₂*Z₁Z₃)/4
    
    # For s₁s₂ = 01
    probs[(0,1)] = (1 + Z₁Z₂ - Z₁Z₃ - Z₁Z₂*Z₁Z₃)/4
    
    # For s₁s₂ = 10
    probs[(1,0)] = (1 - Z₁Z₂ + Z₁Z₃ - Z₁Z₂*Z₁Z₃)/4
    
    # For s₁s₂ = 11
    probs[(1,1)] = (1 - Z₁Z₂ - Z₁Z₃ + Z₁Z₂*Z₁Z₃)/4
    
    # Handle numerical errors
    for k in keys(probs)
        probs[k] = max(0.0, min(1.0, real(probs[k])))
    end
    
    # Normalize probabilities
    total = sum(values(probs))
    for k in keys(probs)
        probs[k] /= total
    end
    
    return probs
end

# Function to display syndrome measurement probabilities
function display_syndrome_probabilities(probabilities::Dict{Tuple{Int,Int}, Float64})
    println("\nSyndrome Measurement Probabilities:")
    for ((s1, s2), prob) in sort(collect(probabilities))
        println("P(s₁s₂ = $s1$s2) = $(round(prob, digits=4))")
    end
end

# Modified measure_syndromes! function
function measure_syndromes!(ψ::MPS)
    # Calculate probabilities
    probs = compute_syndrome_probabilities(ψ)
    display_syndrome_probabilities(probs)
    
    # Choose syndrome based on probabilities
    r = rand()
    cumulative_prob = 0.0
    chosen_syndrome = (0, 0)
    
    for ((s1, s2), prob) in sort(collect(probs))
        cumulative_prob += prob
        if r <= cumulative_prob
            chosen_syndrome = (s1, s2)
            break
        end
    end
    
    println("\nMeasured syndrome: s₁s₂ = $(chosen_syndrome[1])$(chosen_syndrome[2])")
    
    return chosen_syndrome[1], chosen_syndrome[2], ψ
end

# Test function
function test_repetition_code(α::Number, β::Number, γ::Float64)
    # Initialize and encode
    ψ_initial = initialize_state(α, β)
    println("Initial state:")
    display_mps_state(ψ_initial)
    println("\nExpectation values for initial state:")
    println("Z₁Z₂ = $(round(expectation_ZZ(ψ_initial, 1, 2), digits=4))")
    println("Z₁Z₃ = $(round(expectation_ZZ(ψ_initial, 1, 3), digits=4))")
    
    ψ_encoded = encode_state(ψ_initial)
    println("\nEncoded state:")
    display_mps_state(ψ_encoded)
    println("\nExpectation values for encoded state:")
    println("Z₁Z₂ = $(round(expectation_ZZ(ψ_encoded, 1, 2), digits=4))")
    println("Z₁Z₃ = $(round(expectation_ZZ(ψ_encoded, 1, 3), digits=4))")
    
    # Apply error
    ψ_with_error = apply_amplitude_damping(ψ_encoded, 2, γ)
    println("\nState after amplitude damping error (γ = $γ):")
    display_mps_state(ψ_with_error)
    println("\nExpectation values after error:")
    println("Z₁Z₂ = $(round(expectation_ZZ(ψ_with_error, 1, 2), digits=4))")
    println("Z₁Z₃ = $(round(expectation_ZZ(ψ_with_error, 1, 3), digits=4))")
    
    # Compute and display syndrome probabilities
    s1, s2, ψ_measured = measure_syndromes!(ψ_with_error)
    
    return ψ_measured
end

ψ_final = test_repetition_code(0.5, 0.5, 0.5)
