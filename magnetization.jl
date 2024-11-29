using ITensors
using Plots

function create_tfim_hamiltonian(N::Int, h::Float64)
    # Create array of sites
    sites = siteinds("S=1/2", N)
    
    # Initialize Hamiltonian MPO
    ampo = OpSum()
    
    # Add ZZ interaction terms with periodic boundary condition
    for j in 1:N
        ampo += -1.0, "Z", j, "Z", mod1(j+1, N)
    end
    
    # Add transverse field terms
    for j in 1:N
        ampo += -h, "X", j
    end
    
    # Convert to MPO
    H = MPO(ampo, sites)
    
    return H, sites
end

function get_ground_state(H, sites)
    # Random initial state
    psi0 = randomMPS(sites)
    
    # DMRG sweeps parameters
    nsweeps = 20  # Increased number of sweeps for larger system
    maxdim = [10,20,100,200,300,400,500]
    cutoff = 1E-12  # Tighter cutoff
    
    # Find ground state
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    
    return psi
end

function compute_magnetization(psi, sites)
    # Compute total magnetization in z direction
    mz = 0.0
    for i in 1:length(sites)
        # Measure local z magnetization for each site
        ops = OpSum()
        ops += 1.0, "X", i
        op_mpo = MPO(ops, sites)
        local_mz = real(inner(psi', op_mpo, psi))
        mz += local_mz
    end
    
    # Return average magnetization
    return mz / length(sites)
end

# Compute magnetization as a function of h
function magnetization_vs_h(N::Int, h_values::StepRangeLen{Float64})
    magnetizations = Float64[]
    
    for h in h_values
        # Create Hamiltonian and find ground state
        H, sites = create_tfim_hamiltonian(N, h)
        psi = get_ground_state(H, sites)
        
        # Compute magnetization
        mz = compute_magnetization(psi, sites)
        push!(magnetizations, abs(mz))
    end
    
    return magnetizations
end

# Parameters
h_values = range(0.0, 2.0, length=50)  # Range of magnetic field strengths
system_sizes = [32, 64, 128, 256]  # Different system sizes to explore

# Create plot
p = plot(xlabel="Magnetic Field Strength (h)", 
         ylabel="Average Magnetization ⟨Mx⟩",
         title="Magnetization vs Magnetic Field for Different System Sizes",
         legend=:topleft,
         linewidth=2,
         size=(800, 600))

# Compute and plot magnetization for each system size
for N in system_sizes
    magnetizations = magnetization_vs_h(N, h_values)
    plot!(p, h_values, magnetizations, label="N = $N")
end

# Save the plot
savefig(p, "tfim_magnetization_multiple_sizes.png")

# Print the results for verification
println("System Sizes: ", system_sizes) 