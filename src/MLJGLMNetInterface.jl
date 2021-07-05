module MLJGLMNetInterface

export ElasticNetRegressor, ElasticNetCVRegressor

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

const EN_DESCR = "Elastic Net with defaults `alpha = 1` (LASSO) and `family = Normal()`"
const ENCV_DESCR = "Cross-Validated Elastic Net with optimal `lambda` and " *
        "defaults `alpha = 1` (LASSO) and `family = Normal()`"

####
#### REGRESSION TYPES
####

struct ElasticNetRegressor{F, K} <: MMI.Deterministic
    family::F
    kwargs::K
end
ElasticNetRegressor(; family = Distributions.Normal(), kwargs...) = ElasticNetRegressor(family, kwargs)
struct ElasticNetCVRegressor{F, K} <: MMI.Deterministic
    family::F
    kwargs::K
end
ElasticNetCVRegressor(; family = Distributions.Normal(), kwargs...) = ElasticNetCVRegressor(family, kwargs)

###
## Helper functions
###

"""
glmnet_report(model, features, fitresult)

Report based on the `fitresult` of a GLMNet model.
"""
function glmnet_report(::ElasticNetRegressor, features, fitresult)
    (features = features,
     nactive = GLMNet.nactive(fitresult.betas),
     dev_ratio = fitresult.dev_ratio,
     lambda = fitresult.lambda)
end
function glmnet_report(::ElasticNetCVRegressor, features, fitresult)
    x, i = findmin(fitresult.meanloss)
    (features = features,
     lambdamin = fitresult.lambda[i],
     meanloss = x,
     std = fitresult.stdloss[i])
end

####
#### FIT FUNCTIONS
####

glmnet(model::ElasticNetRegressor, X, y) = GLMNet.glmnet(X, y, model.family; model.kwargs...)
glmnet(model::ElasticNetCVRegressor, X, y) = GLMNet.glmnetcv(X, y, model.family; model.kwargs...)

function MMI.fit(model::Union{ElasticNetRegressor, ElasticNetCVRegressor}, verbosity::Int, X, y)
    # apply the model
    features  = Tables.schema(X).names
    Xmatrix   = MMI.matrix(X)
    fitresult = glmnet(model, Xmatrix, y)
    # form the report
    report    = glmnet_report(model, features, fitresult)
    cache     = nothing
    # return
    return fitresult, cache, report
end

function MMI.fitted_params(::ElasticNetRegressor, fitresult)
    (coef = fitresult.betas, intercept = fitresult.a0)
end
function MMI.fitted_params(model::ElasticNetCVRegressor, fitresult)
    ind = argmin(fitresult.meanloss)
    if isa(model.family, Distributions.Multinomial)
        (coef = fitresult.path.betas[:, :, ind], intercept = fitresult.path.a0[:, ind])
    else
        (coef = fitresult.path.betas[:, ind], intercept = fitresult.path.a0[ind])
    end
end

function MMI.predict(model::Union{ElasticNetRegressor, ElasticNetCVRegressor}, fitresult, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    GLMNet.predict(fitresult, Xmatrix)
end

####
#### METADATA
####

# shared metadata
metadata_pkg.((ElasticNetRegressor, ElasticNetCVRegressor),
              name       = "GLMNet",
              uuid       = "8d5ece8b-de18-5317-b113-243142960cc6",
              url        = "https://github.com/JuliaStats/GLMNet.jl",
              julia      = true,
              license    = "MIT",
              is_wrapper = true
              )

metadata_model(ElasticNetRegressor,
               input   = Table(Continuous),
               target  = AbstractVecOrMat{Continuous},
               weights = true,
               descr   = EN_DESCR,
               path    = "$PKG.ElasticNetRegressor"
               )

metadata_model(ElasticNetCVRegressor,
               input   = Table(Continuous),
               target  = AbstractVecOrMat{Continuous},
               weights = true,
               descr   = ENCV_DESCR,
               path    = "$PKG.ElasticNetCVRegressor"
               )

end # module
