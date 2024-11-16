using ITensors

# Define the site indices
N = 5  # Logical + Ancilla + Measurement qubits
s = siteinds("S=1/2", N)

# Custom operator definitions for the gates and projectors
function ITensors.op(::OpName"CX", ::SiteType"S=1/2", s1::Index, s2::Index)
    mat = [1 0 0 0
           0 1 0 0
           0 0 0 1
           0 0 1 0]
    return ITensor(mat, s2', s1', s2, s1)
end

function ITensors.op(::OpName"X", ::SiteType"S=1/2", s::Index)
    mat = [0 1
           1 0]
    return ITensor(mat, s', s)
end

# Projectors for measurement outcomes
function ITensors.op(::OpName"Π0", ::SiteType"S=1/2", s::Index)
    mat = [1 0
           0 0]
    return ITensor(mat, s', s)
end

function ITensors.op(::OpName"Π1", ::SiteType"S=1/2", s::Index)
    mat = [0 0
           0 1]
    return ITensor(mat, s', s)
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

# Encode the logical qubit state across the 3-qubit repetition code
function encode_state(ψ::MPS)
    encoding_ops = [
        ("CX", 1, 2),
        ("CX", 1, 3)
    ]
    ψ_encoded = apply(ops(encoding_ops, s), ψ)
    return normalize!(ψ_encoded)
end

# Function to apply an error (bit-flip X gate) on a specific qubit
function apply_error(ψ::MPS, site::Int)
    return apply(ops([("X", site)], s), ψ)
end

# Measure Z1Z2 and Z1Z3 syndromes using ancilla qubits
function measure_syndromes!(ψ::MPS)
    ψ_meas = copy(ψ)
    
    # Apply CNOT gates to transfer syndrome to ancilla qubit 4
    ψ_meas = apply(ops([("CX", 1, 4), ("CX", 2, 4)], s), ψ_meas)
    outcome_12, ψ_meas = measure_site!(ψ_meas, 4)
    
    # Apply CNOT gates to transfer syndrome to ancilla qubit 5
    ψ_meas = apply(ops([("CX", 1, 5), ("CX", 3, 5)], s), ψ_meas)
    outcome_13, ψ_meas = measure_site!(ψ_meas, 5)
    
    return outcome_12, outcome_13
end

# Function to measure a single qubit (ancilla) and collapse its state
function measure_site!(ψ::MPS, site::Int)
    ψ = orthogonalize!(ψ, site)
    ϕ = ψ[site]
    ρ = prime(ϕ, tags="Site") * dag(ϕ)
    prob = real.(diag(array(ρ)))
    outcome = (rand() < prob[1] ? 0 : 1)
    ψ = apply(ops([("Π$outcome", site)], s), ψ)
    normalize!(ψ)
    return outcome, ψ
end

# Perform error correction by identifying the syndrome and applying necessary correction
function error_correction!(ψ::MPS)
    s1, s2 = measure_syndromes!(ψ)
    
    if s1 == 1 && s2 == 0
        println("Error detected on qubit 2")
        ψ = apply_error(ψ, 2)
    elseif s1 == 0 && s2 == 1
        println("Error detected on qubit 3")
        ψ = apply_error(ψ, 3)
    elseif s1 == 1 && s2 == 1
        println("Error detected on qubit 1")
        ψ = apply_error(ψ, 1)
    else
        println("No error detected")
    end
    return ψ
end

# Decode the repetition code back to the original logical qubit
function decode_state(ψ::MPS)
    decoding_ops = [
        ("CX", 1, 3),
        ("CX", 1, 2)
    ]
    ψ_decoded = apply(ops(decoding_ops, s), ψ)
    normalize!(ψ_decoded)
    return ψ_decoded
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

# Test the 3-qubit repetition code with an arbitrary input state
function test_repetition_code(α::Number, β::Number)
    # Step 1: Initialize arbitrary state |ψ> = α|0⟩ + β|1⟩
    ψ_initial = initialize_state(α, β)
    println("Initial state prepared.")
    display_mps_state(ψ_initial)
    
    # Step 2: Encode the state into the repetition code
    ψ_encoded = encode_state(ψ_initial)
    println("State encoded.")
    display_mps_state(ψ_encoded)
    
    # Step 3: Introduce an error on qubit 2
    ψ_with_error = apply_error(ψ_encoded, 1)
    println("Bit-flip error applied on qubit 2.")
    display_mps_state(ψ_with_error)

    # Step 4: Perform error correction
    ψ_corrected = error_correction!(ψ_with_error)
    println("Error correction performed.")
    display_mps_state(ψ_corrected)

    # Step 5: Decode the state back to the original qubit
    ψ_final = decode_state(ψ_corrected)
    println("State decoded.")
    display_mps_state(ψ_final)
    # Measure the logical qubit to observe final state
    outcome, ψ_final = measure_site!(ψ_final, 1)
    println("Final logical qubit measurement outcome: |$outcome⟩")
    
    return ψ_final
end

# Example usage: arbitrary initial state |ψ⟩ = (|0⟩ + |1⟩)/√2
ψ_final = test_repetition_code(0.13892, 0.6297412)
