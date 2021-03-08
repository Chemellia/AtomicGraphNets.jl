using Documenter, AtomicGraphNets

makedocs(
	sitename = "AtomicGraphNets.jl",
	modules = [AtomicGraphNets],
	pages = Any[
		"Home" => "index.md",
		"GCNNs" => "gcnns.md",
	]
)
deploydocs(
	repo = "github.com/thazhemadam/AtomicGraphNets.jl.git",
	target = "build",
	branch = "gh-pages"
)
