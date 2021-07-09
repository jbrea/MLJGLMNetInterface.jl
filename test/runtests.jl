using MLJGLMNetInterface, MLJBase, Statistics, Test, Tables
import GLMNet, Distributions, StableRNGs

rng(seed = 123) = StableRNGs.StableRNG(seed)
const RNG = rng(123)

X = [1:100 (1:100)+randn(RNG, 100)*5 (1:100)+randn(RNG, 100)*10 (1:100)+randn(RNG, 100)*20];
y = collect(1:100) + randn(RNG, 100)*10;
tX = table(X)

mach = machine(ElasticNetRegressor(), tX, y) |> fit!
path = GLMNet.glmnet(X, y)
@test fitted_params(mach).coef == path.betas
@test fitted_params(mach).intercept == path.a0

mach = machine(ElasticNetCVRegressor(rng = rng(17), nfolds = 10), tX, y) |> fit!
res = GLMNet.glmnetcv(X, y, rng = rng(17), nfolds = 10)
@test fitted_params(mach).coef == GLMNet.coef(res)
@test report(mach).lambdamin == res.lambda[argmin(res.meanloss)]

y = collect(1:100) + rand(RNG, 1:20, 100);
mach = machine(ElasticNetCVCountRegressor(rng = rng()), tX, y) |> fit!
res = GLMNet.glmnetcv(X, y, Distributions.Poisson(), rng = rng())
@test report(mach).minmeanloss == minimum(res.meanloss)

y = rand(RNG, 1:10, 100, 2)
mach = machine(ElasticNetCVClassifier(rng = rng()), tX, y) |> fit!
res = GLMNet.glmnetcv(X, y, Distributions.Binomial(), rng = rng())
@test report(mach).minmeanloss == minimum(res.meanloss)
@test fitted_params(mach).coef == GLMNet.coef(res)

y = coerce(rand(("bli", "bla"), 100), Multiclass)
mach = machine(ElasticNetCVClassifier(rng = rng()), tX, y) |> fit!
res = GLMNet.glmnetcv(X, string.(y), rng = rng())
@test report(mach).minmeanloss == minimum(res.meanloss)
@test fitted_params(mach).coef == GLMNet.coef(res)

y = rand(RNG, 1:10, 100, 3)
mach = machine(ElasticNetCVClassifier(rng = rng()), tX, y) |> fit!
res = GLMNet.glmnetcv(X, y, Distributions.Multinomial(), rng = rng())
@test report(mach).minmeanloss == minimum(res.meanloss)
@test fitted_params(mach).coef == GLMNet.coef(res)

y = coerce(rand(("bli", "bla", "blu"), 100), Multiclass)
mach = machine(ElasticNetCVClassifier(rng = rng()), tX, y) |> fit!
res = GLMNet.glmnetcv(X, string.(y), rng = rng())
@test report(mach).minmeanloss == minimum(res.meanloss)
@test fitted_params(mach).coef == GLMNet.coef(res)
