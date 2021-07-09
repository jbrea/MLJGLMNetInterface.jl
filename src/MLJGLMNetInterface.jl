module MLJGLMNetInterface

export ElasticNetRegressor, ElasticNetCVRegressor,
       ElasticNetCountRegressor, ElasticNetCVCountRegressor,
       ElasticNetClassifier, ElasticNetCVClassifier
# TODO: ElasticNetCoxRegressor, ElasticNetCVCoxRegressor

import MLJModelInterface
import MLJModelInterface: metadata_pkg, metadata_model,
                          Table, Continuous, Count, Finite, OrderedFactor,
                          Multiclass, @mlj_model
import Distributions
using Tables
import GLMNet
using Parameters

const MMI = MLJModelInterface
const PKG = "MLJGLMNetInterface"

##
## DESCRIPTIONS
##

const EN_DESCR = "Elastic Net with defaults `alpha = 1` (LASSO)"
const ENCV_DESCR = "Cross-Validated Elastic Net with optimal `lambda` and " *
        "defaults `alpha = 1` (LASSO)"

####
#### REGRESSION TYPES
####

struct ElasticNetRegressor{K} <: MMI.Probabilistic
    kwargs::K
end
ElasticNetRegressor(; kwargs...) = ElasticNetRegressor(kwargs)
struct ElasticNetCVRegressor{K} <: MMI.Probabilistic
    kwargs::K
end
ElasticNetCVRegressor(; kwargs...) = ElasticNetCVRegressor(kwargs)
struct ElasticNetCountRegressor{K} <: MMI.Probabilistic
    kwargs::K
end
ElasticNetCountRegressor(; kwargs...) = ElasticNetCountRegressor(kwargs)
struct ElasticNetCVCountRegressor{K} <: MMI.Probabilistic
    kwargs::K
end
ElasticNetCVCountRegressor(; kwargs...) = ElasticNetCVCountRegressor(kwargs)
struct ElasticNetClassifier{K} <: MMI.Probabilistic
    kwargs::K
end
ElasticNetClassifier(; kwargs...) = ElasticNetClassifier(kwargs)
struct ElasticNetCVClassifier{K} <: MMI.Probabilistic
    kwargs::K
end
ElasticNetCVClassifier(; kwargs...) = ElasticNetCVClassifier(kwargs)
# struct ElasticNetCoxRegressor{K} <: MMI.Probabilistic
#     kwargs::K
# end
# ElasticNetCoxRegressor(; kwargs...) = ElasticNetCoxRegressor(kwargs)
# struct ElasticNetCVCoxRegressor{K} <: MMI.Probabilistic
#     kwargs::K
# end
# ElasticNetCVCoxRegressor(; kwargs...) = ElasticNetCVCoxRegressor(kwargs)

const ElasticNets = Union{ElasticNetRegressor,
                          ElasticNetCountRegressor,
                          ElasticNetClassifier,
#                           ElasticNetCoxRegressor,
                         }
const ElasticNetsCV = Union{ElasticNetCVRegressor,
                            ElasticNetCVCountRegressor,
                            ElasticNetCVClassifier,
#                             ElasticNetCVCoxRegressor,
                           }
###
## Helper functions
###

"""
glmnet_report(model, features, fitresult)

Report based on the `fitresult` of a GLMNet model.
"""
function glmnet_report(::ElasticNets, features, fitresult)
    (
     lambda = fitresult.lambda,
     dev_ratio = fitresult.dev_ratio,
     nactive = GLMNet.nactive(fitresult.betas),
     features = [features...],
    )

end
function glmnet_report(::ElasticNetsCV, features, fitresult)
    x, i = findmin(fitresult.meanloss)
    (
     lambdamin = fitresult.lambda[i],
     minmeanloss = x,
     meanloss = fitresult.meanloss,
     lambda = fitresult.lambda,
     std = fitresult.stdloss[i],
     features = [features...],
    )
end

####
#### FIT FUNCTIONS
####

family(model, ::Any) = family(model)
family(::Union{ElasticNetRegressor, ElasticNetCVRegressor}) = Distributions.Normal()
family(::Union{ElasticNetCountRegressor, ElasticNetCVCountRegressor}) = Distributions.Poisson()
# family(::Union{ElasticNetCoxRegressor, ElasticNetCVCoxRegressor}) = Distributions.CoxPH()
function family(::Union{ElasticNetClassifier, ElasticNetCVClassifier}, y)
    n_classes = isa(y, AbstractMatrix) ? size(y, 2) : length(MMI.classes(y))
    if n_classes == 2
        Distributions.Binomial()
    else
        Distributions.Multinomial()
    end
end

plain(y) = plain(MMI.scitype(y), y)
plain(::Any, y) = y
plain(::Type{<:AbstractVector{<:Finite}}, y) = convert(Matrix{Float64}, [i == j for i in y, j in MMI.classes(y)])

glmnet(model::ElasticNets, X, y) = GLMNet.glmnet(X, plain(y), family(model, y); model.kwargs...)
glmnet(model::ElasticNetsCV, X, y) = GLMNet.glmnetcv(X, plain(y), family(model, y); model.kwargs...)

function MMI.fit(model::Union{ElasticNets, ElasticNetsCV}, verbosity::Int, X, y)
    # apply the model
    features  = Tables.schema(X).names
    Xmatrix   = MMI.matrix(X)
    fitresult = glmnet(model, Xmatrix, y)
    # form the report
    report    = glmnet_report(model, features, fitresult)
    cache     = nothing
    # return
    return (fitresult, y), cache, report
end

function MMI.fitted_params(::ElasticNets, (fitresult, _))
    (coef = fitresult.betas, intercept = fitresult.a0)
end
function MMI.fitted_params(model::ElasticNetsCV, (fitresult, decode))
    ind = argmin(fitresult.meanloss)
    if isa(family(model, decode), Distributions.Multinomial)
        (coef = fitresult.path.betas[:, :, ind], intercept = fitresult.path.a0[:, ind])
    else
        (coef = fitresult.path.betas[:, ind], intercept = fitresult.path.a0[ind])
    end
end

function MMI.predict_mean(::Union{ElasticNets, ElasticNetsCV}, (fitresult, _), Xnew)
    Xmatrix = MMI.matrix(Xnew)
    GLMNet.predict(fitresult, Xmatrix, outtype = :notlink)
end

deviance(r, null_dev, n) = null_dev * (1 - r) / (n - 2)
function deviance(::ElasticNetRegressor, fitresult, n)
    deviance.(fitresult.dev_ratio, fitresult.null_dev, n)
end
function deviance(::ElasticNetCVRegressor, fitresult, n)
    i = argmin(fitresult.meanloss)
    [deviance(fitresult.path.dev_ratio[i], fitresult.path.null_dev, n)]
end

function _predict(model::Union{ElasticNetRegressor, ElasticNetCVRegressor},
                  η, (fitresult, decode))
    σ = deviance(model, fitresult, length(decode))
    [Distributions.Normal(η[i, j], σ[j]) for i in 1:size(η, 1), j in 1:size(η, 2)]
end

function _predict(::Union{ElasticNetClassifier, ElasticNetCVClassifier},
                  η, (_, decode))
    cls = isa(decode, AbstractMatrix) ? (1:size(decode, 2)) : MMI.classes(decode)
    if length(cls) == 2
        η = [1 .- η η]
    end
    MMI.UnivariateFinite(cls, η)
end

function _predict(::Union{ElasticNetCountRegressor, ElasticNetCVCountRegressor},
                  η, ::Any)
    Distributions.Poisson.(η)
end

function MMI.predict(model::Union{ElasticNets, ElasticNetsCV}, fitresult, Xnew)
    η = MMI.predict_mean(model, fitresult, Xnew)
    _predict(model, η, fitresult)
end

####
#### METADATA
####

# shared metadata
metadata_pkg.((ElasticNetRegressor, ElasticNetCVRegressor,
               ElasticNetCountRegressor, ElasticNetCVCountRegressor,
               ElasticNetClassifier, ElasticNetCVClassifier,
#                ElasticNetCoxRegressor, ElasticNetCVCoxRegressor,
              ),
              name       = "GLMNet",
              uuid       = "8d5ece8b-de18-5317-b113-243142960cc6",
              url        = "https://github.com/JuliaStats/GLMNet.jl",
              julia      = true,
              license    = "MIT",
              is_wrapper = true
              )

metadata_model(ElasticNetRegressor,
               input   = Table(Continuous),
               target  = AbstractVector{Continuous},
               weights = true,
               descr   = EN_DESCR,
               path    = "$PKG.ElasticNetRegressor"
               )

metadata_model(ElasticNetCVRegressor,
               input   = Table(Continuous),
               target  = AbstractVector{Continuous},
               weights = true,
               descr   = ENCV_DESCR,
               path    = "$PKG.ElasticNetCVRegressor"
               )

metadata_model(ElasticNetCountRegressor,
               input   = Table(Continuous),
               target  = AbstractVector{Count},
               weights = true,
               descr   = EN_DESCR,
               path    = "$PKG.ElasticNetCountRegressor"
               )

metadata_model(ElasticNetCVCountRegressor,
               input   = Table(Continuous),
               target  = AbstractVector{Count},
               weights = true,
               descr   = ENCV_DESCR,
               path    = "$PKG.ElasticNetCVCountRegressor"
               )

metadata_model(ElasticNetClassifier,
               input   = Table(Continuous),
               target  = Union{AbstractVector{<:Finite}, AbstractMatrix{<:Count}},
               weights = true,
               descr   = EN_DESCR,
               path    = "$PKG.ElasticNetClassifier"
               )

metadata_model(ElasticNetCVClassifier,
               input   = Table(Continuous),
               target  = Union{AbstractVector{<:Finite}, AbstractMatrix{<:Count}},
               weights = true,
               descr   = ENCV_DESCR,
               path    = "$PKG.ElasticNetCVClassifier"
               )

# metadata_model(ElasticNetCoxRegressor,
#                input   = Table(Continuous),
#                target  = AbstractMatrix{Continuous},
#                weights = true,
#                descr   = EN_DESCR,
#                path    = "$PKG.ElasticNetCoxRegressor"
#                )
#
# metadata_model(ElasticNetCVCoxRegressor,
#                input   = Table(Continuous),
#                target  = AbstractMatrix{Continuous},
#                weights = true,
#                descr   = ENCV_DESCR,
#                path    = "$PKG.ElasticNetCVCoxRegressor"
#                )
end # module
