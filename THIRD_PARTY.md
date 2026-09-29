# Third-party source notices

This file records upstream notices for third-party components only; it does not grant a license for the surrounding project.

- `anchoring.py` retains a Python reference of the exact-quote Hypothesis text path, pinned to [Hypothesis client b4d085a2](https://github.com/hypothesis/client/tree/b4d085a2f893aa6de3b61d8b8bc3ae4d0f24fc1a). The minimum-substring-edit score reproduces the scalar score consumed from [approx-string-match-js d45a0bfd](https://github.com/robertknight/approx-string-match-js/tree/d45a0bfd65a4f1a0f8daebc5e75a4f6f0aea0cdd); it uses the existing Python dynamic program, not the upstream JavaScript implementation. Both upstream notices are retained below. The port is limited to exact-quote-preserved flat text and is not a DOM implementation.
- `vendor/reanchor/` contains five unmodified Python runtime files from [microbluey/reanchor b91143e7](https://github.com/microbluey/reanchor/tree/b91143e7f3f48e9356a8bec846da39a2527ab04d), covered by `vendor/reanchor/LICENSE`. The adapter checks their pinned hashes. Upstream package metadata declared 0.3.0; the unchanged runtime version string is 0.1.0. Commit identity governs the comparison.

## Hypothesis client

Exact-commit license source: https://raw.githubusercontent.com/hypothesis/client/b4d085a2f893aa6de3b61d8b8bc3ae4d0f24fc1a/LICENSE

```text
Copyright (c) 2013-2019 Hypothes.is Project and contributors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Subcomponents

The project includes a number of subcomponents with separate
copyright notices and license terms. Your use of the code for the these
subcomponents is subject to the terms and conditions of the following licenses.

For the annotator subcomponent:

 Copyright 2012 Aron Carroll, Rufus Pollock, and Nick Stenning.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
```

## approx-string-match-js

Exact-commit license source: https://raw.githubusercontent.com/robertknight/approx-string-match-js/d45a0bfd65a4f1a0f8daebc5e75a4f6f0aea0cdd/LICENSE

```text
MIT License

Copyright (c) 2020 Robert Knight

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
