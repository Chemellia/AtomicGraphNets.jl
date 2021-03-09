using Documenter, AtomicGraphNets

makedocs(
	sitename = "AtomicGraphNets.jl",
	modules = [AtomicGraphNets],
	pages = Any[
		"Home" => "index.md",
		"Basic Graph Theory" => "graph_theory.md",
		"GCNNs" => "gcnns.md",
		"Comparison with cgcnn.py" => "comparison.md",
	]
)
deploydocs(
	repo = "github.com/thazhemadam/AtomicGraphNets.jl.git",
	target = "build",
	branch = "gh-pages"
)
