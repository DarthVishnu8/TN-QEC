using ITensors

# Define the site indices for 7 qubits
N = 7
s = siteinds("S=1/2", N)

# Add X and Z operator definitions
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



# Initialize logical state |ψ⟩ = α|0⟩ + β|1⟩
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

# Encode into Steane code
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

# Function to create identity plus/minus Pauli operator for a single site
function create_site_projector(operator_type::String, sign::Int, s::Index)
    if operator_type == "Z"
        # (I ± Z)/2 = [1±1 0; 0 1∓1]/2
        return ITensor([1.0+sign 0.0; 0.0 1.0-sign]/2.0, s', s)
    else # X
        # (I ± X)/2 = [1 ±1; ±1 1]/2
        return ITensor([1.0 sign; sign 1.0]/2.0, s', s)
    end
end

# Function to compute syndrome probabilities correctly with MPS structure
function compute_syndrome_probabilities(ψ::MPS)
    probs = Dict{Tuple{Int,Int,Int,Int,Int,Int}, Float64}()
    
    # Define the stabilizer operators and their locations
    stabilizers = [
        ("X", [1,2,3,4]),  # X1234
        ("X", [1,2,5,6]),  # X1256
        ("X", [1,3,5,7]),  # X1357
        ("Z", [1,2,3,4]),  # Z1234
        ("Z", [1,2,5,6]),  # Z1256
        ("Z", [1,3,5,7])   # Z1357
    ]
    
    # Iterate through all possible syndrome patterns
    for s1 in 0:1, s2 in 0:1, s3 in 0:1, s4 in 0:1, s5 in 0:1, s6 in 0:1
        syndromes = [s1, s2, s3, s4, s5, s6]
        
        # Make a copy of the state to work with
        ψ_projected = copy(ψ)
        
        # Apply each stabilizer projector
        for (idx, (op_type, qubits)) in enumerate(stabilizers)
            sign = (-1)^syndromes[idx]
            
            # Apply projector operators site by site
            for q in qubits
                ψ_projected = orthogonalize!(ψ_projected, q)
                projector = create_site_projector(op_type, sign, s[q])
                ψ_projected[q] = projector * ψ_projected[q]
            end
        end
        
        # Compute probability
        prob = abs2(inner(ψ, ψ_projected))
        probs[(s1,s2,s3,s4,s5,s6)] = max(0.0, real(prob))
    end
    
    # Normalize probabilities
    total = sum(values(probs))
    for k in keys(probs)
        probs[k] /= total
    end
    
    return probs
end

# Function to apply amplitude damping error
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

# Function to display syndrome probabilities
function display_syndrome_probabilities(probs::Dict{Tuple{Int,Int,Int,Int,Int,Int}, Float64})
    println("\nSyndrome Measurement Probabilities:")
    for (syndrome, prob) in sort(collect(probs))
        if prob > 0.001  # Only show non-negligible probabilities
            println("P($(syndrome)) = $(round(prob, digits=4))")
        end
    end
end

# Test function
function test_steane_code(α::Number, β::Number, γ::Float64, error_site::Int)
    # Initialize and encode
    ψ_initial = initialize_state(α, β)
    println("Initial state prepared")
    display_mps_state(ψ_initial)

    ψ_encoded = encode_steane(ψ_initial)
    println("State encoded using Steane code")
    display_mps_state(ψ_encoded)

    # Apply error
    ψ_with_error = apply_amplitude_damping(ψ_encoded, error_site, γ)
    println("\nApplied amplitude damping error with γ = $γ on qubit $error_site")
    display_mps_state(ψ_with_error)

    # Compute and display syndrome probabilities
    probs = compute_syndrome_probabilities(ψ_with_error)
    display_syndrome_probabilities(probs)
    
    return ψ_with_error
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


# Run test with interesting parameters
ψ_final = test_steane_code(0.0, 1.0, 0.0, 2)