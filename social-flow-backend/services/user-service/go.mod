module user-service

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/aws/aws-sdk-go-v2 v1.21.0
    github.com/aws/aws-sdk-go-v2/service/cognitoidentityprovider v1.37.0
    github.com/aws/aws-sdk-go-v2/service/kms v1.28.0
    github.com/redis/go-redis/v9 v9.0.5
    golang.org/x/crypto v0.14.0
    // more dependencies for robust backend
)
