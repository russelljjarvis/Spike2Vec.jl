

using CairoMakie

noto_sans = "../assets/NotoSans-Regular.ttf"
noto_sans_bold = "../assets/NotoSans-Bold.ttf"

fig = Figure(backgroundcolor = RGBf0(0.98, 0.98, 0.98),
    resolution = (1000, 700), font = noto_sans)

ax1 = fig[1, 1] = Axis(fig, title = "Pre Treatment")

data1 = randn(50, 2) * [1 2.5; 2.5 1] .+ [10 10]

line1 = lines!(ax1, 5..15, x -> x, color = :red, linewidth = 2)
scat1 = scatter!(ax1, data1,
    color = (:red, 0.3), markersize = 15px, marker = '■')

ax2, line2 = lines(fig[1, 2], 7..17, x -> -x + 26,
    color = :blue, linewidth = 2,
    axis = (title = "Post Treatment",))

data2 = randn(50, 2) * [1 -2.5; -2.5 1] .+ [13 13]

scat2 = scatter!(data2,
    color = (:blue, 0.3), markersize = 15px, marker = '▲')

linkaxes!(ax1, ax2)

hideydecorations!(ax2, grid = false)

ax1.xlabel = "Weight [kg]"
ax2.xlabel = "Weight [kg]"
ax1.ylabel = "Maximum Velocity [m/sec]"

leg = fig[1, end+1] = Legend(fig,
    [line1, scat1, line2, scat2],
    ["f(x) = x", "Data", "f(x) = -x + 26", "Data"])

fig[2, 1:2] = leg

trim!(fig.layout)

leg.tellheight = true

leg.orientation = :horizontal

hm_axes = fig[1:2, 3] = [Axis(fig, title = t) for t in ["Cell Assembly Pre", "Cell Assembly Post"]]

heatmaps = [heatmap!(ax, i .+ rand(20, 20)) for (i, ax) in enumerate(hm_axes)]

hm_sublayout = GridLayout()
fig[1:2, 3] = hm_sublayout

# there is another shortcut for filling a GridLayout vertically with
# a vector of content
hm_sublayout[:v] = hm_axes

hidedecorations!.(hm_axes)

for hm in heatmaps
    hm.colorrange = (1, 3)
end

cbar = hm_sublayout[:, 2] = Colorbar(fig, heatmaps[1], label = "Activity [spikes/sec]")

cbar.height = Relative(2/3)

cbar.ticks = 1:0.5:3

supertitle = fig[0, :] = Label(fig, "Complex Figures with Makie",
    textsize = 24, font = noto_sans_bold, color = (:black, 0.25))

label_a = fig[2, 1, TopLeft()] = Label(fig, "A", textsize = 24,
    font = noto_sans_bold, halign = :right)
label_b = fig[2, 3, TopLeft()] = Label(fig, "B", textsize = 24,
    font = noto_sans_bold, halign = :right)

label_a.padding = (0, 6, 16, 0)
label_b.padding = (0, 6, 16, 0)

# Aspect(1, 1) means that relative to row 1
# (row because we're setting a colsize,
# and aspect ratios are always about the other side)
# we set the column to an aspect ratio of 1

colsize!(hm_sublayout, 1, Aspect(1, 1))

save("layout_tutorial_final.svg", fig)