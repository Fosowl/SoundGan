# Go Backend
FROM golang:latest

WORKDIR /app

COPY . .

# Build the Go application
RUN go mod tidy && go build -o main .

# Expose the backend port
EXPOSE 8080

# Start the Go server
CMD ["./main"]