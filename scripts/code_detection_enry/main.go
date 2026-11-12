// Reads one JSON string per line on stdin; prints go-enry's language guess per line,
// from the content alone (no filename), with the classifier's candidates over all languages.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/go-enry/go-enry/v2"
	"github.com/go-enry/go-enry/v2/data"
)

func main() {
	var all []string
	for l := range data.LanguagesLogProbabilities {
		all = append(all, l)
	}
	sort.Strings(all)
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1<<24), 1<<24)
	for in.Scan() {
		var text string
		if err := json.Unmarshal(in.Bytes(), &text); err != nil {
			panic(err)
		}
		lang, safe := enry.GetLanguageByContent("", []byte(text))
		if lang == "" {
			lang, safe = enry.GetLanguageByClassifier([]byte(text), all)
		}
		fmt.Printf("%s\t%v\n", lang, safe)
	}
}
