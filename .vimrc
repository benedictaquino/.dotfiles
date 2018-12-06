" My vimrc file
" Maintainer: Benedict Aquino
" Last change: 2018-11-21

" Get the defaults
source $VIMRUNTIME/defaults.vim

if has("vms")
    set nobackup       " do not keep a backup file, use versions instead
else
    set backup          " keep a backup file (restore to previous version)
    if has("persistent_undo")
        set undofile    " keep an undo file (undo changes after closing)
    endif
endif

" Only do this part when compiled with support for autocommands
if has("autocmd")
    " Put these in an autocmd group, so that we can delete them easily
    augroup vimrcEx
        au!
        " For all text files set 'textwidth' to 78 characters
        autocmd FileType text setlocal textwidth=78
    augroup END
else
    set autoindent      " always set autoindenting on
endif

if has("syntax") && has("eval")
    packadd! matchit
endif

"Easier split navigations
nnoremap <C-J> <C-W><C-J>
nnoremap <C-K> <C-W><C-K>
nnoremap <C-L> <C-W><C-L>
nnoremap <C-H> <C-W><C-H>

" More natural split opening
set splitbelow
set splitright

" Spaces not tabs
set tabstop=4
set shiftwidth=4
set expandtab

" Show line numbers
set number

execute pathogen#infect()
syntax enable
set background=dark

colorscheme solarized

let g:airline_theme="solarized"
let g:airline_solarized_bg="dark"
let g:airline_powerline_fonts=1
let g:airline#extensions#tabline#enabled = 1

set rtp+=~/.fzf
