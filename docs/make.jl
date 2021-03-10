using Documenter, AtomicGraphNets

# The DOCSARGS environment variable can be used to pass additional arguments to make.jl.
# This is useful on CI, if you need to change the behavior of the build slightly but you
# can not change the .travis.yml or make.jl scripts any more (e.g. for a tag build).
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

makedocs(
	sitename = "AtomicGraphNets.jl",
	modules = [AtomicGraphNets],
	pages = Any[
		"Home" => "index.md",
		"Basic Graph Theory" => "graph_theory.md",
		"GCNNs" => "gcnns.md",
		"Comparison with cgcnn.py" => "comparison.md",
	],
	format = Documenter.HTML(
		# Use clean URLs, unless built as a "local" build
		prettyurls = !("local" in ARGS),
		canonical = "https://thazhemadam.github.io/AtomicGraphNets.jl/stable/",	
	),
	linkcheck = "linkcheck" in ARGS,
)
deploydocs(
	repo = "github.com/thazhemadam/AtomicGraphNets.jl.git",
	target = "build",
	branch = "gh-pages",
	push_preview = true
)