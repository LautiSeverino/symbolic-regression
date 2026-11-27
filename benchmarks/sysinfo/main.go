// sysinfo prints Go runtime environment information for use by benchmark
// scripts. Output format: KEY=VALUE, one entry per line.
//
// Usage (from repository root):
//
//	go build -o /tmp/sysinfo ./benchmarks/sysinfo && /tmp/sysinfo
package main

import (
	"fmt"
	"runtime"
)

func main() {
	fmt.Printf("GOMAXPROCS=%d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("NumCPU=%d\n", runtime.NumCPU())
	fmt.Printf("GoVersion=%s\n", runtime.Version())
	fmt.Printf("GOOS=%s\n", runtime.GOOS)
	fmt.Printf("GOARCH=%s\n", runtime.GOARCH)
}
