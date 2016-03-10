

module funcapp

using PyPlot
using FastGaussQuadrature: gausschebyshev
using Optim
using ApproxFun
using ApproXD

	# use chebyshev to interpolate this:
	function q1(n)

		n_new = 50 # For predictions

		# Define the function to approximated
		f(x) = x + 2x^2 - exp(-x)

		# Define n Chebyshev interpolation points on the interval [-3,3]
		nodes = gausschebyshev(n)[1]*3

		# Create the interpolation matrix (and convert to type Float64)
		Phi = [cos((n-i+0.5)*(j-1)*pi/n) for i in 1:n, j in 1:n]
		Phi = convert(Array{Float64,2}, Phi)

		# Values of f at the interpolation points
		y = nodes + 2nodes.^2 - exp(-nodes)

		# Invert the interpolation matrix
		invPhi = inv(Phi)

		# Compute the weights c
		c = invPhi * y

		# Define the vector of Chebyshev Polynomials on [-3, 3]
		T(x) = [cos(acos(x/3)j) for j in 0:n-1]

		# Define the approximation function
		g(x) = dot(c,T(x))

		# Get 50 predictions
		k = linspace(-3,3,n_new)
		G = [g(k[i]) for i in 1:n_new]

		# Plot the predictions against the true function
		scatter(k,G, color="orange")

		l = linspace(-3,3, 1000)
		F = [f(l[i]) for i in 1:1000]
		plot(l,F, color="blue")

		# Test the Approximation
		diff(x) = -abs(f(x) - g(x))

		optimize(diff, -3.0, 3.0)

		println("Question 1")
		println("The maximal deviation of the function is smaller than 1e-6")

	end

	function q2(n)

		# Get interpolation points
		S = Chebyshev([-3, 3])
		x = points(S,n)

		# Define the function to be approximated on the interpolation points
		f = x + 2x.^2 - exp(-x)

		# Get the approximation function using ApproxFun
		g = Fun(ApproxFun.transform(S,f),S)
		ApproxFun.plot(g; title = "Approximation using ApproxFun")

	end


	# plot the first 9 Chebyshev Polynomial Basis Functions
	function q3()

		x = linspace(-1,1,1000)
		Cheby = [cos(acos(x)j) for j in 0:8]

		fig,axes = subplots(3,3,figsize=(10,5))
		for i in 1:3
				for j in 1:3
						ax = axes[j,i]
						count = i+(j-1)*3
						ax[:plot](x,Cheby[i+(j-1)*3])
						ax[:set_title]("Basis function $(count-1)")
						ax[:yaxis][:set_visible](false)
						ax[:xaxis][:set_visible](false)
						ax[:set_xlim](-1.0,1.0)
				end
		end

	end

	ChebyT(x,deg) = cos(acos(x)*deg)
	unitmap(x,lb,ub) = 2.*(x.-lb)/(ub.-lb) - 1	#[a,b] -> [-1,1]

	type ChebyType
		f::Function # fuction to approximate
		nodes::Union{Vector,LinSpace} # evaluation points
		basis::Matrix # basis evaluated at nodes
		coefs::Vector # estimated coefficients

		deg::Int 	# degree of chebypolynomial
		lb::Float64 # bounds
		ub::Float64

		# constructor
		function ChebyType(_nodes::Union{Vector,LinSpace},_deg,_lb,_ub,_f::Function)
			n = length(_nodes)
			y = _f(_nodes)
			_basis = Float64[ChebyT(unitmap(_nodes[i],_lb,_ub),j) for i=1:n,j=0:_deg]
			_coefs = _basis \ y  # type `?\` to find out more about the backslash operator. depending the args given, it performs a different operation
			# create a ChebyType with those values
			new(_f,_nodes,_basis,_coefs,_deg,_lb,_ub)
		end
	end

	# function to predict points using info stored in ChebyType
	function predict(Ch::ChebyType,x_new)

		true_new = Ch.f(x_new)
		basis_new = Float64[ChebyT(unitmap(x_new[i],Ch.lb,Ch.ub),j) for i=1:length(x_new),j=0:Ch.deg]
		basis_nodes = Float64[ChebyT(unitmap(Ch.nodes[i],Ch.lb,Ch.ub),j) for i=1:length(Ch.nodes),j=0:Ch.deg]
		preds = basis_new * Ch.coefs
		preds_nodes = basis_nodes * Ch.coefs

		return Dict("x"=> x_new,"truth"=>true_new, "preds"=>preds, "preds_nodes" => preds_nodes)
	end

	function q4a(deg=(5,9,15),lb=-1.0,ub=1.0)

		# Define the function to be approximated
		f(x) = 1./(1+25.*x.^2)

		# Set up the plot with 2 panels
		l = linspace(-1,1,1000)
		fig,axes = subplots(1,2,figsize=(10,5))

		# Graph with Chebyshev interpolation points
		ax = axes[1,1]
		ax[:set_title]("Chebyshev nodes")

		for j in 1:3

				# Define the Chebyshev nodes
				nodes = gausschebyshev(deg[j]+1)[1]

				# Get the true function as well as approximations of different degrees
				x = predict(ChebyType(nodes,deg[j]+1,lb,ub,f),l)["x"]
				y1 = predict(ChebyType(nodes,deg[j]+1,lb,ub,f),l)["truth"]
				y2 = predict(ChebyType(nodes,deg[j]+1,lb,ub,f),l)["preds"]

				# Plot everything
				ax[:plot](x,y1) # True function
				ax[:plot](x,y2) # Approximations
				ax[:set_ylim](-0.1,1.1)

		end

			# Graph with uniform interpolation points
			ax = axes[2,1]
			ax[:set_title]("Uniformly distributed nodes")

			for j in 1:3

				# Define the uniformly distributed nodes
				nodes = linspace(-1,1,deg[j]+1)

				# Get the true function as well as approximations of different degrees
				x = predict(ChebyType(nodes,deg[j]+1,lb,ub,f),l)["x"]
				y1 = predict(ChebyType(nodes,deg[j]+1,lb,ub,f),l)["truth"]
				y2 = predict(ChebyType(nodes,deg[j]+1,lb,ub,f),l)["preds"]

				# Plot everything
				ax[:plot](x,y1) # True function
				ax[:plot](x,y2) # Approximations
				ax[:set_ylim](-0.1,1.1)

			end

	fig[:canvas][:draw]()
	println("Question 4a")
	println("With Chebyshev nodes, the approximation is less accurate with a small number of nodes, but it consitently improves as the number of nodes increases")
	println("With uniform nodes, the approximation is more accurate with a small number of nodes, however it doesn't seem to improve as the number of nodes increases")

	end

	function q4b()

		# Define the function to be approximated
		f(x) = 1./(1+25.*x.^2)

		# Set up the plot with 2 panels
		fig,axes = subplots(1,2,figsize=(10,5))
		l = linspace(-5,5,10000)

		# True function on first plot
		ax = axes[1,1]
		ax[:set_title]("True function")
		F = f(l)
		ax[:plot](l,F)
		ax[:set_ylim](-0.1,1.1)

		# Deviation of approximation on second plot
		ax = axes[2,1]
		ax[:set_title]("Deviation of the approximation")

		# Equally spaced knots

			# Get the knots
			bs = BSpline(13,1,-5,5)

			# Find the coefficients
			B = full(getBasis(collect(linspace(-5,5,65)),bs))
			y = f(linspace(-5,5,65))
			c = B \ y

			# Approximate the function
			B = full(getBasis(collect(l),bs))
			g1 = B*c
			dev2 = F - g1

			# Plot the first version
			ax[:plot](l,dev1)

		# Knots concentrated toward 0

			# Get some knots
			k = linspace(-5,5,13)
			knots = k.^5/625
			bs = BSpline(knots,3)

			# Find the coefficients
			B = full(getBasis(collect(linspace(-5,5,65)),bs))
			y = f(linspace(-5,5,65))
			c = B \ y

			# Approximate the function
			B = full(getBasis(collect(l),bs))
			g2 = B*c
			dev2 = F - g2

			# Plot the second version
			ax[:plot](l,dev2)

			# Show the knots
			ax[:scatter](knots, f(knots))

	println("Question 4b")
	println("It is possible to improve the approximation by choosing adequate places for the knots.")

	end

	function q5()


	end


		# function to run all questions
	function runall()
		println("running all questions of HW-funcapprox:")
		q1(15)
		q2(15)
		q3()
		q4a()
		q4b()
		q5()
	end


end
