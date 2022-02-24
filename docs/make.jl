using Documenter
using AtomicGraphNets

# The DOCSARGS environment variable can be used to pass additional arguments to make.jl.
# This is useful on CI, if you need to change the behavior of the build slightly but you
# can not change the .travis.yml or make.jl scripts any more (e.g. for a tag build).
if haskey(ENV, "DOCSARGS")
    for arg in split(ENV["DOCSARGS"])
        (arg in ARGS) || push!(ARGS, arg)
    end
end

pages = Any[
    "Home"=>"index.md",
    "Basic Graph Theory"=>"graph_theory.md",
    "GCNNs"=>"gcnns.md",
    "Comparison with cgcnn.py"=>"comparison.md",
    "Examples"=>Any[
        "Example 1"=>"examples/example_1.md",
    ],
    "Functions"=>Any[
        "Layers"=>"functions/layers.md",
        "Models"=>"functions/models.md"
    ],
    "Changelog"=>"changelog.md",
]

makedocs(
    sitename = "AtomicGraphNets.jl",
    modules = [AtomicGraphNets],
    pages = pages,
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = !("local" in ARGS),
        canonical = "https://chemellia.github.io/AtomicGraphNets.jl/stable/",
        edit_link = "main",
    ),
    linkcheck = "linkcheck" in ARGS,
)
deploydocs(
    repo = "github.com/Chemellia/AtomicGraphNets.jl.git",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)

# for local build
"""
makedocs(
    sitename = "AtomicGraphNets.jl",
    modules = [AtomicGraphNets],
    pages = pages,
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = false,
        canonical = "https://chemellia.github.io/AtomicGraphNets.jl/stable/",
        edit_link = "main",
    ),
    linkcheck = "linkcheck" in ARGS,
)
"""
