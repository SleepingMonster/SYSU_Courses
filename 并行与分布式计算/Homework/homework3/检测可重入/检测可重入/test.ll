; ModuleID = 'test.c'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%union.pthread_attr_t = type { i64, [48 x i8] }

@Global = common global i32 0, align 4

; Function Attrs: norecurse nounwind uwtable
define noalias i8* @Thread1(i8* nocapture readnone %x) #0 {
  %1 = load i32, i32* @Global, align 4, !tbaa !1
  %2 = add nsw i32 %1, 1
  store i32 %2, i32* @Global, align 4, !tbaa !1
  ret i8* null
}

; Function Attrs: norecurse nounwind uwtable
define noalias i8* @Thread2(i8* nocapture readnone %x) #0 {
  %1 = load i32, i32* @Global, align 4, !tbaa !1
  %2 = add nsw i32 %1, -1
  store i32 %2, i32* @Global, align 4, !tbaa !1
  ret i8* null
}

; Function Attrs: nounwind uwtable
define i32 @main() #1 {
  %t = alloca [2 x i64], align 16
  %1 = bitcast [2 x i64]* %t to i8*
  call void @llvm.lifetime.start(i64 16, i8* %1) #5
  %2 = getelementptr inbounds [2 x i64], [2 x i64]* %t, i64 0, i64 0
  %3 = call i32 @pthread_create(i64* %2, %union.pthread_attr_t* null, i8* (i8*)* nonnull @Thread1, i8* null) #5
  %4 = getelementptr inbounds [2 x i64], [2 x i64]* %t, i64 0, i64 1
  %5 = call i32 @pthread_create(i64* %4, %union.pthread_attr_t* null, i8* (i8*)* nonnull @Thread2, i8* null) #5
  %6 = load i64, i64* %2, align 16, !tbaa !5
  %7 = call i32 @pthread_join(i64 %6, i8** null) #5
  %8 = load i64, i64* %4, align 8, !tbaa !5
  %9 = call i32 @pthread_join(i64 %8, i8** null) #5
  call void @llvm.lifetime.end(i64 16, i8* %1) #5
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #2

; Function Attrs: nounwind
declare i32 @pthread_create(i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*) #3

declare i32 @pthread_join(i64, i8**) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #2

attributes #0 = { norecurse nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0-2ubuntu4 (tags/RELEASE_380/final)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"long", !3, i64 0}
