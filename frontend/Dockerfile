FROM node:18-alpine

# Install dependencies
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Disable telemetry
ENV NEXT_TELEMETRY_DISABLED 1

# Expose port
EXPOSE 3000

# Start development server with host binding for Docker
CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0"]
